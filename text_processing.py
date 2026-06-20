import logging
import re
import time
import numpy as np

from config import JD_EMBEDDING_TEXT

log = logging.getLogger("ranker")

# TEXT BUILDING FOR BM25
 
def build_candidate_text(c: dict) -> str:
    """
    Build a single text blob per candidate for BM25 indexing.
    Emphasizes career descriptions and skills (substance > metadata).
    """
    parts = []
 
    profile = c.get("profile", {})
    # Headline + summary: high signal
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("current_industry", ""))
 
    # Career history: very important
    for ch in c.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
        parts.append(ch.get("industry", ""))
        parts.append(ch.get("company", ""))  # company name matters for product vs consulting
 
    # Skills list
    for sk in c.get("skills", []):
        name = sk.get("name", "")
        prof = sk.get("proficiency", "")
        duration = sk.get("duration_months", 0)
        # Repeat high-proficiency, long-duration skills to give them more weight in BM25
        if prof in ("expert", "advanced") and duration >= 12:
            parts.extend([name] * 3)
        elif prof in ("expert", "advanced"):
            parts.extend([name] * 2)
        else:
            parts.append(name)
 
    # Certifications
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
 
    return " ".join(p for p in parts if p).lower()
 
 
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z][a-z0-9_\-\./]*", text.lower())

# SEMANTIC SCORING  (20% of final score — applied to BM25 top-5K only)
 
def compute_semantic_scores(
    top_indices: list[int],
    texts: list[str],
    use_embeddings: bool = True,
) -> dict[int, float]:
    """
    Compute cosine similarity between JD embedding and candidate texts.
    Only called on BM25 top-5K to stay within time budget.
    Falls back gracefully if sentence-transformers not available.
    """
    if not use_embeddings:
        return {}
 
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("sentence-transformers not installed; skipping semantic scoring.")
        return {}
 
    log.info(f"Computing semantic scores for {len(top_indices):,} candidates ...")
    t0 = time.time()
 
    model = SentenceTransformer("all-MiniLM-L6-v2")
 
    # JD embedding
    jd_emb = model.encode(JD_EMBEDDING_TEXT, convert_to_numpy=True, normalize_embeddings=True)
 
    # Candidate embeddings in batches
    candidate_texts = [texts[i] for i in top_indices]
    embs = model.encode(
        candidate_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
 
    # Cosine similarity (both normalized, so dot product = cosine)
    scores = embs @ jd_emb  # shape (N,)
    scores = np.clip((scores + 1) / 2, 0, 1)  # scale from [-1,1] → [0,1]
 
    result = {idx: float(scores[i]) for i, idx in enumerate(top_indices)}
    log.info(f"Semantic scoring done in {time.time() - t0:.1f}s")
    return result