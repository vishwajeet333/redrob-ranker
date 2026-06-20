import gradio as gr
import json, gzip, csv, tempfile, os
from pathlib import Path
from rank_bm25 import BM25Okapi
import sys
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    build_candidate_text, tokenize, is_honeypot,
    get_career_fit_score, get_skill_fit_score,
    get_behavioral_score, generate_reasoning,
    JD_QUERY_TOKENS
)
import numpy as np

def load_from_bytes(file_path):
    p = Path(file_path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(p) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]

def run_ranking(file_obj):
    if file_obj is None:
        return None, None, "Please upload a candidates file."

    try:
        candidates = load_from_bytes(file_obj)
    except Exception as e:
        return None, None, f"Could not read file: {e}"

    candidates = candidates[:500]  # sandbox cap

    texts     = [build_candidate_text(c) for c in candidates]
    tokenized = [tokenize(t) for t in texts]
    bm25      = BM25Okapi(tokenized)
    scores_bm = bm25.get_scores(JD_QUERY_TOKENS)
    bm_max    = scores_bm.max() or 1.0
    bm_norm   = scores_bm / bm_max

    pool_idx = list(np.argsort(scores_bm)[::-1][:min(200, len(candidates))])

    scored = []
    for idx in pool_idx:
        c   = candidates[idx]
        cid = c.get("candidate_id", f"UNK_{idx}")
        if is_honeypot(c):
            scored.append({"candidate_id": cid, "score": 0.001,
                           "title": c.get("profile",{}).get("current_title","—"),
                           "company": c.get("profile",{}).get("current_company","—"),
                           "yoe": c.get("profile",{}).get("years_of_experience",0),
                           "reasoning": "Honeypot detected."})
            continue

        cs, cd = get_career_fit_score(c)
        ss, _  = get_skill_fit_score(c)
        bs, bd = get_behavioral_score(c)
        bm     = float(bm_norm[idx])

        raw = cs*0.35 + ss*0.25 + bs*0.15 + bm*0.05 + 0.20*0.5
        if cd.get("entirely_consulting"): raw *= 0.35
        if cd.get("wrong_domain") and not cd.get("ai_title_hit") and not cd.get("swe_hit"):
            raw *= 0.15

        final = round(min(1.0, max(0.0, raw)), 4)
        reasoning = generate_reasoning(c, cd, {}, bd, final)
        scored.append({
            "candidate_id": cid,
            "score": final,
            "title":   c.get("profile",{}).get("current_title","—"),
            "company": c.get("profile",{}).get("current_company","—"),
            "yoe":     c.get("profile",{}).get("years_of_experience",0),
            "reasoning": reasoning,
        })

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top100 = scored[:100]

    # Build display table
    table = []
    for rank, r in enumerate(top100[:20], 1):
        table.append([rank, r["candidate_id"], f"{r['score']:.3f}",
                      r["title"], r["company"], f"{r['yoe']:.0f} yrs",
                      r["reasoning"][:80] + "..."])

    headers = ["Rank", "Candidate ID", "Score", "Title", "Company", "YoE", "Reasoning"]

    # Write CSV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w",
                                      newline="", encoding="utf-8")
    writer = csv.writer(tmp)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    prev = float("inf")
    for rank, r in enumerate(top100, 1):
        s = min(r["score"], prev); prev = s
        writer.writerow([r["candidate_id"], rank, f"{s:.6f}",
                         r["reasoning"].replace('"', "'")])
    tmp.close()

    msg = (f"Ranked {len(candidates)} candidates → showing top 20 of 100.\n"
           f"Top score: {top100[0]['score']:.3f}  |  "
           f"Bottom score: {top100[-1]['score']:.3f}")

    return {"headers": headers, "data": table}, tmp.name, msg


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("""
# Redrob Intelligent Candidate Ranker
**Role:** Senior AI Engineer : Founding Team, Redrob AI

Multi-signal ranking: career trajectory + skill quality + behavioral availability.
Upload `sample_candidates.json` or any JSONL file (max 500 candidates in sandbox).
""")

    with gr.Row():
        file_input = gr.File(
            label="Upload candidates file",
            file_types=[".json", ".jsonl", ".gz"],
        )
        run_btn = gr.Button("Rank candidates", variant="primary", scale=0)

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        table_out = gr.Dataframe(
            label="Top 20 ranked candidates",
            wrap=True,
            interactive=False,
        )

    csv_out = gr.File(label="Download full submission.csv (top 100)")

    gr.Markdown("""
---
### How scoring works
| Signal | Weight | What it catches |

| Career trajectory | 35% | Consulting-only penalty ×0.35 · product company bonus |
                
| Skill fit | 25% | Trust-weighted by endorsements + duration, not just listed |
                
| Behavioral signals | 15% | Recency · response rate · notice period · location |
                
| Semantic similarity | 20% | BM25 + JD embedding alignment |
                
| Honeypot detection | hard gate | Expert skills with 0 duration → score ~ 0 |
""")

    run_btn.click(
        fn=run_ranking,
        inputs=[file_input],
        outputs=[table_out, csv_out, status],
    )

if __name__ == "__main__":
    demo.launch()