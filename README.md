<div align="center">

```
██████╗ ███████╗██████╗ ██████╗  ██████╗ ██████╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
██████╔╝█████╗  ██║  ██║██████╔╝██║   ██║██████╔╝
██╔══██╗██╔══╝  ██║  ██║██╔══██╗██║   ██║██╔══██╗
██║  ██║███████╗██████╔╝██║  ██║╚██████╔╝██████╔╝
╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝
         INTELLIGENT CANDIDATE RANKER
```

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Live-FFD21E?style=flat-square)](https://huggingface.co/spaces/holyn/redrob-ranker)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![CPU Only](https://img.shields.io/badge/Inference-CPU%20Only-EF4444?style=flat-square)]()
[![No Network](https://img.shields.io/badge/Ranking-Zero%20API%20Calls-8B5CF6?style=flat-square)]()

**Redrob AI Hackathon · Intelligent Candidate Discovery & Ranking**

*Ranks 100,000 candidates the way a world-class recruiter would —
by understanding career trajectories, not counting keywords.*

[**🚀 Live Demo**](https://huggingface.co/spaces/holyn/redrob-ranker) · [**📄 Quick Start**](#quick-start) · [**🏗 Architecture**](#architecture)

</div>

---

## The core insight

Every ATS ranks candidates by keyword frequency. Ours doesn't.

A candidate who lists **"FAISS, Pinecone, Elasticsearch"** in their skills but spent their entire career at TCS ranks **lower** than someone who shipped a recommendation system at Swiggy and never wrote those words. That distinction — between claimed skills and evidenced work — is what separates this system from a glorified `grep`.

---

## Architecture

```
100,000 candidates (JSONL)
         │
         ▼
┌─────────────────────┐
│  Stage 1 · BM25     │  JD-derived query -> top 5,000 in ~30s
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2 · Embed    │  all-MiniLM-L6-v2 cosine sim (CPU, batched)
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│            Multi-Signal Scorer               │
│                                              │
│  career_fit    · 0.35  ──  domain + company  │
│                             type + prod. ev. │
│                          + title progression │
│  skill_fit     · 0.25  ──  trust-weighted:   │
│                             proficiency ×    │
│                             endorsements ×   │
│                             duration ×       │
│                             career CV factor │
│  semantic      · 0.20  ──  JD cosine sim     │
│  behavioral    · 0.15  ──  recency · notice  │
│                             · response rate  │
│  bm25          · 0.05  ──  lexical signal    │
│                                              │
│  ┌─ hard gates ──────────────────────────┐   │
│  │ consulting-only career   →  × 0.35   │   │
│  │ wrong domain             →  × 0.15   │   │
│  │ honeypot signals         →  ≈ 0.00   │   │
│  └───────────────────────────────────────┘   │
└──────────────┬───────────────────────────────┘
               │
               ▼
     Top 100 · submission.csv · audit_log.jsonl
```

> **Optional:** run `precompute.py` offline to blend Gemini Flash scores as a 25% signal — zero network calls during ranking.

---

## What it catches that keyword rankers miss

| Candidate type | Keyword ranker | This system |
|---|---|---|
| TCS engineer · lists FAISS + Pinecone | ✅ Ranks high | ❌ Consulting-only penalty ×0.35 |
| Swiggy SWE · shipped recommender · no "RAG" | ❌ Ranks low | ✅ Career fit = 1.0 |
| Claims "expert in PyTorch" · used for 0 months | ✅ Full credit | ❌ Cross-validation → trust ×0.5 |
| Inactive 7 months · 4% response rate | Ignored | ❌ Behavioral penalty |
| Expert in 12 skills · all 0 endorsements | ✅ Passes | ❌ Honeypot gate → score ≈ 0 |
| SWE → Senior → Staff in 5 years | Ignored | ✅ Title progression bonus |

---

## Quick start

```bash
# Install
pip install -r requirements.txt

# Rank (standard — ~2-3 min, CPU only, no network)
python main.py --candidates candidates.jsonl.gz --out submission.csv

# Rank without semantic scoring (faster · ~30s)
python main.py --candidates candidates.jsonl --out submission.csv --no-embeddings

# Validation
python validation.py submission.csv
```

---

## Repository layout

```
redrob-ranker/
├── main.py              # Entry point
├── scoring.py           # All scoring logic (career, skill, behavioral)
├── text_processing.py   # BM25 tokenizer + candidate text builder
├── data_io.py           # Loads .json / .jsonl / .jsonl.gz
├── config.py            # JD-derived constants, keyword lists, weights
├── precompute.py        # Offline Gemini pre-computation (run once)
├── app.py               # Gradio sandbox (HuggingFace Spaces)
├── validation.py        # Submission format validator
└── requirements.txt
```

---

## Compute constraints 

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~2–3 min |
| RAM | ≤ 16 GB | ~3–5 GB |
| GPU | ✗ not allowed | ✗ not used |
| Network during ranking | ✗ not allowed | **zero API calls** |

---

## Scoring signals

**Career Fit (35%)** : the highest weight by JD design

- Consults keyword lists for company type (product vs. consulting)
- Counts production deployment signals in role descriptions (`deployed`, `shipped`, `serving`, `A/B test`, `latency` …)
- Scores seniority trajectory — ascending titles across roles get a bonus

**Skill Fit (25%)** : quality-verified, not just listed

- Every skill is weighted `proficiency × endorsements × duration_months`
- Cross-validated against career descriptions — claimed but never evidenced = trust ×0.5
- Redrob assessment scores override claimed proficiency when available
- Zero duration + zero endorsements = keyword stuffing signal -> trust ≈ 0.1

**Behavioral (15%)** : is this candidate actually reachable?

- Days since last active · recruiter response rate · notice period · open-to-work flag · location match (Pune, Noida, Bengaluru, Hyderabad, Mumbai, Delhi)

**Honeypot detection** : hard gate before scoring

- Expert proficiency + 0 `duration_months` across multiple skills -> blocked
- Career timeline discrepancies beyond 24 months -> blocked

---

<div align="center">

Built for the **Redrob AI Hackathon** · Intelligent Candidate Discovery & Ranking Challenge

</div>