import csv
import gzip
import json
import logging
from pathlib import Path

log = logging.getLogger("ranker")

# DATA LOADING
 
def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    candidates = []
 
    log.info(f"Loading candidates from {p} ...")
 
    # Gzipped JSONL
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning(f"Skipping malformed line {i+1}")
                if (i + 1) % 10_000 == 0:
                    log.info(f"  Loaded {i+1:,} ...")
        log.info(f"Loaded {len(candidates):,} candidates from gzipped JSONL")
        return candidates
 
    # Plain file: first JSONL then JSON array
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read().strip()
 
    # If it starts with '[' it's a JSON array (e.g. sample_candidates.json)
    if raw.startswith("["):
        candidates = json.loads(raw)
        log.info(f"Loaded {len(candidates):,} candidates from JSON array")
        return candidates
 
    # Otherwise parse line-by-line as JSONL
    for i, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning(f"Skipping malformed line {i+1}")
        if (i + 1) % 10_000 == 0:
            log.info(f"  Loaded {i+1:,} ...")
 
    log.info(f"Loaded {len(candidates):,} candidates from JSONL")
    return candidates

# AUDIT TRAIL LOGGER
 
def write_audit_log(results: list[dict], path: str) -> None:
    """
    Write per-candidate audit trail as JSON lines.
    Includes score breakdown + reasoning — useful for Stage 4 manual review defense.
    """
    with open(path, "w", encoding="utf-8") as f:
        for rank_idx, r in enumerate(results, 1):
            audit = {
                "rank": rank_idx,
                "candidate_id": r["candidate_id"],
                "final_score": r["final_score"],
                "score_breakdown": {
                    "career_fit": round(r.get("_career_score", 0), 4),
                    "skill_fit": round(r.get("_skill_score", 0), 4),
                    "behavioral": round(r.get("_behav_score", 0), 4),
                    "semantic": round(r.get("_sem_score", 0), 4),
                    "bm25": round(r.get("_bm25_score", 0), 4),
                },
                "career_details": r.get("_career_details", {}),
                "skill_details": r.get("_skill_details", {}),
                "behavioral_details": r.get("_behav_details", {}),
                "reasoning": r["reasoning"],
            }
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")
    log.info(f"Audit log written to {path}")
 
# CSV OUTPUT
 
def write_submission_csv(results: list[dict], path: str) -> None:
    """
    Write validated CSV exactly matching submission_spec.md format.
    - candidate_id, rank, score, reasoning
    - Scores non-increasing
    - Ranks 1-100 exactly once each
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
 
        prev_score = float("inf")
        for rank_idx, r in enumerate(results, 1):
            score = r["final_score"]
            # Guarantee non-increasing scores (handle floating point edge cases)
            score = min(score, prev_score)
            prev_score = score
            # Escape commas/quotes in reasoning
            reasoning = r["reasoning"].replace('"', "'")
            writer.writerow([r["candidate_id"], rank_idx, f"{score:.6f}", reasoning])
 
    log.info(f"Submission CSV written to {path}")