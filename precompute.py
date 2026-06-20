import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# Import BM25 utilities from rank.py 
sys.path.insert(0, str(Path(__file__).parent))
from main import build_candidate_text, tokenize, JD_QUERY_TOKENS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute")

# GEMINI ASSESSMENT PROMPT

JD_CONTEXT = """
You are a senior technical recruiter evaluating candidates for this role:

ROLE: Senior AI Engineer — Founding Team
COMPANY: Early-stage AI startup (Series A)
CORE REQUIREMENTS:
- 5-9 years total, 4+ years applied ML at product companies (NOT consulting firms)
- Production experience with embeddings-based retrieval (FAISS, Pinecone, Qdrant, etc.)
- Vector databases and hybrid search infrastructure at scale
- Evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing)
- Strong Python, PyTorch, HuggingFace
NICE TO HAVE: LLM fine-tuning (LoRA/QLoRA), learning-to-rank, HR-tech exposure

DISQUALIFIERS:
- Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, etc.)
- Non-technical backgrounds (marketing, operations, finance, civil/mechanical engineering)
- Only research experience with no production deployments
- No evidence of working with retrieval/ranking/recommendation systems
"""

SCORING_PROMPT_TEMPLATE = """{jd_context}

Evaluate this candidate profile and return ONLY a JSON object with no other text:

CANDIDATE:
{candidate_text}

Return exactly:
{{
  "score": <integer 0-100 reflecting fit for this specific role>,
  "reason": "<one sentence: the single most important factor driving your score>",
  "red_flags": ["<flag 1>", "<flag 2>"],
  "strengths": ["<strength 1>", "<strength 2>"]
}}

Scoring guide:
85-100: Near-perfect fit. Right domain, product company, evidence of production retrieval systems.
65-84:  Good fit. Right domain and company type, some retrieval experience, minor gaps.
45-64:  Partial fit. Relevant but missing key requirement (e.g., good SWE but no retrieval work).
25-44:  Weak fit. Wrong domain or consulting-only, but some tangential relevance.
0-24:   Poor fit. Wrong domain, consulting-only, or honeypot/impossible profile.
"""


def build_candidate_summary(c: dict, max_chars: int = 800) -> str:
    """
    Build a concise candidate summary for the Gemini prompt.
    Keeps tokens low while preserving the most signal-rich information.
    """
    profile = c.get("profile", {})
    lines = [
        f"Title: {profile.get('current_title', 'Unknown')}",
        f"Company: {profile.get('current_company', 'Unknown')} "
        f"(size: {profile.get('current_company_size', '?')})",
        f"Years of experience: {profile.get('years_of_experience', '?')}",
        f"Location: {profile.get('location', '?')}, {profile.get('country', '?')}",
        f"Headline: {profile.get('headline', '')}",
        "",
        "Career history:",
    ]

    for ch in c.get("career_history", [])[:4]:  # cap at 4 roles
        lines.append(
            f"  - {ch.get('title', '?')} @ {ch.get('company', '?')} "
            f"({ch.get('duration_months', 0)} months): "
            f"{ch.get('description', '')[:150]}"
        )

    lines.append("")
    lines.append("Skills: " + ", ".join(
        f"{s['name']} ({s.get('proficiency', '?')}, {s.get('duration_months', 0)}mo)"
        for s in c.get("skills", [])[:10]
    ))

    text = "\n".join(lines)
    return text[:max_chars]


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(p, encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]

# GEMINI BATCH SCORING

def score_batch_with_gemini(candidates: list[dict], client, model_name: str) -> dict:
    """
    Score a batch of candidates with Gemini. Returns dict of
    candidate_id -> {"score": int, "reason": str, "red_flags": list, "strengths": list}
    """
    results = {}
    failed = 0

    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", f"UNKNOWN_{i}")
        candidate_text = build_candidate_summary(c)

        prompt = SCORING_PROMPT_TEMPLATE.format(
            jd_context=JD_CONTEXT,
            candidate_text=candidate_text,
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,   # low temp for determinism
                    "max_output_tokens": 256,
                }
            )
            text = response.text.strip()
            clean = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)

            # Validate expected keys
            score = int(result.get("score", 50))
            score = max(0, min(100, score))
            results[cid] = {
                "score":     score,
                "reason":    str(result.get("reason", ""))[:200],
                "red_flags": result.get("red_flags", [])[:3],
                "strengths": result.get("strengths", [])[:3],
            }

        except Exception as e:
            log.warning(f"  [{i+1}] {cid}: Gemini call failed ({e}), skipping")
            failed += 1
            continue

        # Progress logging
        if (i + 1) % 50 == 0:
            log.info(f"  Scored {i+1}/{len(candidates)} | Failed: {failed}")

        # Respect rate limits: Gemini Flash allows ~15 RPM on free tier
        # Adjust sleep based on your tier (0 for paid, 4 for free)
        time.sleep(13.0)

    log.info(f"Batch complete: {len(results)} scored, {failed} failed")
    return results

# MAIN

def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob Offline Gemini Pre-computation")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="gemini_scores.json",
                        help="Output path for Gemini scores (default: gemini_scores.json)")
    parser.add_argument("--bm25-pool", type=int, default=5000,
                        help="How many top BM25 candidates to send to Gemini (default: 5000)")
    parser.add_argument("--model", default="gemini-1.5-flash",
                        help="Gemini model to use (default: gemini-1.5-flash)")
    parser.add_argument("--resume", default=None,
                        help="Path to existing gemini_scores.json to resume from (skip already scored)")
    args = parser.parse_args()

    # API key 
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("Set GEMINI_API_KEY environment variable before running.")
        log.error("  Windows: set GEMINI_API_KEY=your-key")
        log.error("  Linux/Mac: export GEMINI_API_KEY=your-key")
        sys.exit(1)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        log.info(f"Gemini client initialized. Model: {args.model}")
    except ImportError:
        log.error("google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    # Load candidates 
    log.info(f"Loading candidates from {args.candidates} ...")
    t0 = time.time()
    candidates = load_candidates(args.candidates)
    log.info(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # BM25 pre-filter -> top N 
    log.info("Running BM25 to select top candidates for Gemini scoring ...")
    texts     = [build_candidate_text(c) for c in candidates]
    tokenized = [tokenize(t) for t in texts]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(JD_QUERY_TOKENS)
    top_idx   = list(np.argsort(scores)[::-1][:args.bm25_pool])
    top_cands = [candidates[i] for i in top_idx]
    log.info(f"Selected top {len(top_cands):,} candidates for Gemini scoring")

    # Resume support
    existing_scores = {}
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            existing_scores = json.load(f)
        log.info(f"Resuming: {len(existing_scores):,} already scored, loading from {args.resume}")

    to_score = [c for c in top_cands if c.get("candidate_id", "") not in existing_scores]
    log.info(f"Candidates to score: {len(to_score):,} (skipping {len(top_cands)-len(to_score):,} already done)")

    if not to_score:
        log.info("All candidates already scored. Nothing to do.")
        return

    # Score with Gemini 
    log.info(f"Sending {len(to_score):,} candidates to {args.model} ...")
    log.info("Estimated time: ~{:.0f} minutes".format(len(to_score) * 0.2 / 60))
    t_score = time.time()

    new_scores = score_batch_with_gemini(to_score, client, args.model)
    all_scores = {**existing_scores, **new_scores}

    # Save results 
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_score
    log.info(f"Done. {len(new_scores):,} new scores in {elapsed:.0f}s → saved to {args.out}")
    log.info(f"Total scores saved: {len(all_scores):,}")
    log.info("")
    log.info("Next step:")
    log.info(f"  python rank.py --candidates {args.candidates} "
             f"--out submission.csv --gemini-scores {args.out}")


if __name__ == "__main__":
    main()