from datetime import date

# JD-DERIVED CONFIGURATION (Hardcoded from deep JD analysis so no network needed)
 
EVAL_DATE = date(2026, 6, 19)
 
# BM25 query: tokens that represent what this JD actually means (not just says)
JD_QUERY_TOKENS = (
    "embeddings vector search retrieval ranking hybrid search bm25 semantic search "
    "sentence transformers faiss elasticsearch opensearch recommendation system "
    "evaluation framework ndcg mrr map python machine learning nlp information retrieval "
    "llm fine-tuning product company startup scale production deployment "
    "pinecone weaviate qdrant milvus vector database dense retrieval sparse retrieval "
    "reranking learning to rank a/b testing online evaluation offline evaluation "
    "pytorch huggingface transformers information retrieval search system"
).split()
 
# JD text for semantic embedding query
JD_EMBEDDING_TEXT = (
    "Senior AI Engineer at a Series A startup. Must have production experience with "
    "embeddings-based retrieval systems, vector databases, hybrid search infrastructure "
    "(FAISS, Elasticsearch, Pinecone, Milvus, Qdrant). Strong Python. Designed evaluation "
    "frameworks for ranking systems (NDCG, MRR, MAP, A/B testing). LLM fine-tuning "
    "experience (LoRA, QLoRA) preferred. Applied ML at product companies, not consulting. "
    "Shipped end-to-end ranking, recommendation, or search system to real users at scale."
)
 
# Core skills aligned with JD "Things you absolutely need"
CORE_SKILL_KEYWORDS = {
    # Embeddings & semantic search
    "embeddings", "sentence-transformers", "sentence transformers", "dense retrieval",
    "semantic search", "bge", "e5", "text-embedding", "openai embeddings",
    "bi-encoder", "cross-encoder", "colbert", "vector embeddings",
    # Vector DBs & hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "hybrid search", "ann", "pgvector", "chroma", "chromadb",
    "vector database", "vector store", "approximate nearest neighbor",
    # Ranking & retrieval systems
    "ranking system", "recommendation system", "retrieval system", "search system",
    "learning to rank", "lambdarank", "ranknet", "listwise", "reranking",
    "re-ranking", "information retrieval", "candidate ranking",
    # Evaluation
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "evaluation framework", "a/b testing", "offline evaluation", "online evaluation",
    "retrieval evaluation", "ranking evaluation",
    # Core language / frameworks
    "python", "pytorch", "transformers", "huggingface", "hugging face",
}
 
# Good-to-have skills (bonus weight, not required)
BONUS_SKILL_KEYWORDS = {
    "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
    "rlhf", "instruction tuning", "llm", "large language model",
    "nlp", "natural language processing", "xgboost", "lightgbm",
    "distributed systems", "kafka", "spark", "ray", "open source",
    "mlops", "bm25", "tf-idf", "sparse retrieval", "inverted index",
    "rag", "retrieval augmented generation", "recruiting tech", "hr tech",
    "marketplace", "inference optimization", "model serving", "triton",
}
 
# Consulting-only disqualifier: entire career in these = heavy penalty
# JD explicitly says "only consulting in their entire career = not fit"
# but "currently at one but has prior product experience = fine"
CONSULTING_FIRM_PATTERNS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "niit technologies", "mindtree", "l&t infotech",
    "zensar", "persistent", "mastech", "kpit", "cyient",
}
 
# Product companies that are strong positive signals (partial list)
PRODUCT_COMPANY_SIGNALS = {
    "swiggy", "zomato", "ola", "flipkart", "amazon", "google", "microsoft",
    "meta", "netflix", "uber", "airbnb", "stripe", "razorpay", "paytm",
    "phonepe", "meesho", "cred", "slice", "groww", "zepto", "blinkit",
    "freshworks", "zoho", "chargebee", "browserstack", "cleartax",
    "urban company", "urban clap", "oyo", "mmt", "makemytrip",
    "nykaa", "mamaearth", "boat", "byju", "unacademy", "vedantu",
    "mad street den", "slintel", "darwinbox", "leadsquared", "capillary",
    "media.net", "inmobi", "glance", "daily hunt", "sharechat", "moj",
    "dream11", "games24x7", "juspay", "setu", "open financial",
}
 
# AI/ML relevant title keywords
AI_TITLE_KEYWORDS = {
    "ml engineer", "machine learning", "ai engineer", "applied scientist",
    "research scientist", "nlp engineer", "data scientist", "search engineer",
    "ranking engineer", "recommendation engineer", "applied ml",
    "computer vision engineer",  # even CV counts if they have NLP too
}
 
# Soft titles that are NOT relevant (high confidence disqualifiers)
WRONG_DOMAIN_TITLE_KEYWORDS = {
    "civil engineer", "mechanical engineer", "accountant", "marketing manager",
    "operations manager", "customer support", "sales manager",
    "hr manager", "financial analyst", "graphic designer", "content writer",
    "seo", "brand manager", "supply chain", "procurement", "finance manager",
    "chartered accountant", "ca ", "legal", "lawyer",
}
 
# Indian metro cities aligned with JD
TARGET_INDIA_LOCATIONS = {
    "noida", "pune", "bangalore", "bengaluru", "hyderabad", "mumbai",
    "delhi", "gurugram", "gurgaon", "chennai", "new delhi", "delhi ncr",
}