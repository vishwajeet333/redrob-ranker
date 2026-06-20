import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import JD_QUERY_TOKENS
from data_io import load_candidates, write_audit_log, write_submission_csv
from text_processing import build_candidate_text, tokenize, compute_semantic_scores
from scoring import (
    is_honeypot,
    get_career_fit_score,
    get_skill_fit_score,
    get_behavioral_score,
    generate_reasoning
)

# LOGGING
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ranker")

# MAIN RANKING PIPELINE
 
def rank_candidates(
    candidates: list[dict],
    top_n: int = 100,
    bm25_pool: int = 5000,
    use_embeddings: bool = True,
    gemini_scores: dict | None = None,
) -> list[dict]:
    """
    Full ranking pipeline.
    Returns list of top_n candidate dicts with score and reasoning.
 
    gemini_scores: optional dict mapping candidate_id -> {"score": 0-100, "reason": str} 
    Blended in as a 25% weight signal.
    """
    if gemini_scores:
        log.info(f"Gemini pre-computed scores loaded: {len(gemini_scores):,} candidates")
    total = len(candidates)
    log.info(f"Ranking {total:,} candidates → top {top_n}")
 
    # Step 1: Build text corpus

    log.info("Building text corpus ...")
    t0 = time.time()
    texts = [build_candidate_text(c) for c in candidates]
    tokenized = [tokenize(t) for t in texts]
    log.info(f"Corpus built in {time.time()-t0:.1f}s")
 
    # Step 2: BM25 fast filter → top bm25_pool

    log.info("Fitting BM25 ...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(JD_QUERY_TOKENS)
    bm25_max = max(bm25_scores) if bm25_scores.max() > 0 else 1.0
    bm25_norm = bm25_scores / bm25_max  # normalize to [0, 1]
    log.info(f"BM25 scored {total:,} in {time.time()-t0:.1f}s")
 
    # Get top bm25_pool indices
    top_indices = list(np.argsort(bm25_scores)[::-1][:bm25_pool])
    log.info(f"BM25 pool size: {len(top_indices):,}")
 
    # Step 3: Semantic scoring on BM25 pool
    semantic_scores = compute_semantic_scores(top_indices, texts, use_embeddings)
 
    # Step 4: Multi-signal scoring on BM25 pool
    log.info("Scoring candidates ...")
    t0 = time.time()
    scored = []
 
    for idx in top_indices:
        c = candidates[idx]
        cid = c.get("candidate_id", f"UNKNOWN_{idx}")
 
        # Honeypot gate: score near zero immediately
        if is_honeypot(c):
            scored.append({
                "candidate_id": cid,
                "idx": idx,
                "final_score": 0.001,
                "reasoning": "Profile flagged for honeypot signals; excluded.",
                "_honeypot": True,
            })
            continue
 
        career_score, career_d = get_career_fit_score(c)
        skill_score, skill_d   = get_skill_fit_score(c)
        behav_score, behav_d   = get_behavioral_score(c)
        sem_score  = semantic_scores.get(idx, 0.5)
        bm25_s     = float(bm25_norm[idx])
 
        # Gemini offline score
        gemini_d = None
        gemini_s = None
        if gemini_scores:
            gemini_d = gemini_scores.get(cid)
            if gemini_d:
                gemini_s = min(1.0, max(0.0, gemini_d.get("score", 50) / 100.0))
 
        # Composite scoring formula 

        # With Gemini: redistribute 25% to it, reducing semantic and skill weight
        # Without Gemini: standard weights

        if gemini_s is not None:
            raw_score = (
                career_score * 0.30   # reduced: Gemini subsumes some career signal
                + skill_score  * 0.20  # reduced
                + gemini_s     * 0.25  # NEW: deep LLM assessment (offline)
                + sem_score    * 0.10  # reduced
                + behav_score  * 0.12  # slightly reduced
                + bm25_s       * 0.03
            )
        else:
            raw_score = (
                career_score * 0.35
                + skill_score  * 0.25
                + sem_score    * 0.20
                + behav_score  * 0.15
                + bm25_s       * 0.05
            )
 
        # Hard penalties
        if career_d.get("entirely_consulting"):
            raw_score *= 0.35
        if career_d.get("wrong_domain") and not career_d.get("ai_title_hit") and not career_d.get("swe_hit"):
            raw_score *= 0.15
        if behav_d.get("days_inactive", 0) > 180 and behav_d.get("response_rate", 1) < 0.1:
            raw_score *= 0.5
 
        final_score = round(min(1.0, max(0.0, raw_score)), 6)
 
        reasoning = generate_reasoning(c, career_d, skill_d, behav_d, final_score, gemini_d)
 
        scored.append({
            "candidate_id":   cid,
            "idx":            idx,
            "final_score":    final_score,
            "reasoning":      reasoning,
            "_career_score":  career_score,
            "_skill_score":   skill_score,
            "_behav_score":   behav_score,
            "_sem_score":     sem_score,
            "_bm25_score":    bm25_s,
            "_gemini_score":  gemini_s,
            "_career_details": career_d,
            "_skill_details":  skill_d,
            "_behav_details":  behav_d,
        })
 
    log.info(f"Scoring done in {time.time()-t0:.1f}s")
 
    # Step 5: Sort and deduplicate 

    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
 
    # If pool < top_n (e.g. testing with sample data), score remaining candidates
    # with rule-based signals only so the CSV is padded to exactly 100 rows.
    if len(scored) < top_n:
        scored_ids = {r["candidate_id"] for r in scored}
        remaining = [
            (i, c) for i, c in enumerate(candidates)
            if c.get("candidate_id", "") not in scored_ids
        ]
        log.info(f"Pool smaller than {top_n}; scoring {len(remaining)} additional candidates ...")
        for idx, c in remaining:
            cid = c.get("candidate_id", f"UNKNOWN_{idx}")
            career_score, career_d = get_career_fit_score(c)
            skill_score, skill_d = get_skill_fit_score(c)
            behav_score, behav_d = get_behavioral_score(c)
            raw = career_score * 0.40 + skill_score * 0.30 + behav_score * 0.20
            if career_d.get("entirely_consulting"):
                raw *= 0.35
            if career_d.get("wrong_domain") and not career_d.get("ai_title_hit") and not career_d.get("swe_hit"):
                raw *= 0.15
            final_score = round(min(1.0, max(0.0, raw)), 6)
            reasoning = generate_reasoning(c, career_d, skill_d, behav_d, final_score)
            scored.append({
                "candidate_id": cid, "idx": idx, "final_score": final_score,
                "reasoning": reasoning, "_career_score": career_score,
                "_skill_score": skill_score, "_behav_score": behav_score,
                "_sem_score": 0.0, "_bm25_score": 0.0,
                "_career_details": career_d, "_skill_details": skill_d, "_behav_details": behav_d,
            })
        scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
 
    top = scored[:top_n]
 
    if top:
        log.info(
            f"Top {len(top)} score range: "
            f"{top[0]['final_score']:.4f} → {top[-1]['final_score']:.4f}"
        )
    if len(top) < top_n:
        log.warning(
            f"Only {len(top)} candidates available (needed {top_n}). "
            "Submission will fail format validation — run on the full dataset."
        )
    return top
 
# ENTRY POINT
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob Hackathon Candidate Ranker")
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--audit", default="audit_log.jsonl",
        help="Path for per-candidate audit trail (default: audit_log.jsonl)"
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip semantic scoring (faster, less accurate)"
    )
    parser.add_argument(
        "--bm25-pool", type=int, default=5000,
        help="BM25 pre-filter pool size (default: 5000)"
    )
    parser.add_argument(
        "--gemini-scores", default=None,
        help="Path to gemini_scores.json produced by precompute.py (optional, improves quality)"
    )
    args = parser.parse_args()
 
    t_start = time.time()
 
    # Gemini pre-computed scores
    gemini_scores = None
    if args.gemini_scores:
        gpath = Path(args.gemini_scores)
        if gpath.exists():
            with open(gpath) as f:
                gemini_scores = json.load(f)
            log.info(f"Loaded Gemini scores for {len(gemini_scores):,} candidates from {gpath}")
        else:
            log.warning(f"--gemini-scores path not found: {gpath}. Proceeding without.")
 
    candidates = load_candidates(args.candidates)
    if not candidates:
        log.error("No candidates loaded. Exiting.")
        sys.exit(1)
 
    results = rank_candidates(
        candidates,
        top_n=100,
        bm25_pool=args.bm25_pool,
        use_embeddings=not args.no_embeddings,
        gemini_scores=gemini_scores,
    )
 
    write_submission_csv(results, args.out)
    write_audit_log(results, args.audit)
 
    elapsed = time.time() - t_start
    log.info(f"Total runtime: {elapsed:.1f}s")
    if elapsed > 300:
        log.warning("Runtime exceeded 5 minutes — check compute constraints!")
 
 
if __name__ == "__main__":
    main()