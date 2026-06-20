import re
from datetime import datetime

from config import (
    EVAL_DATE,
    CORE_SKILL_KEYWORDS,
    BONUS_SKILL_KEYWORDS,
    CONSULTING_FIRM_PATTERNS,
    PRODUCT_COMPANY_SIGNALS,
    AI_TITLE_KEYWORDS,
    WRONG_DOMAIN_TITLE_KEYWORDS,
    TARGET_INDIA_LOCATIONS
)

# SKILL-TO-CAREER CROSS-VALIDATION
 
def cross_validate_skill(skill_name: str, career_history: list[dict]) -> float:
    """
    Returns a cross-validation factor (0.5-1.0) based on whether the claimed
    skill actually appears anywhere in the candidate's career descriptions/titles.

    """
    if not career_history:
        return 0.75
 
    all_career_text = " ".join(
        (ch.get("description", "") + " " + ch.get("title", "") + " " + ch.get("company", ""))
        for ch in career_history
    ).lower()
 
    skill_lower = skill_name.lower()
 
    if skill_lower in all_career_text:
        return 1.0
 
    tokens = [t for t in re.findall(r"[a-z][a-z0-9\-]+", skill_lower) if len(t) > 3]
    if tokens and any(t in all_career_text for t in tokens):
        return 0.85
 
    return 0.5  # skill claimed but never evidenced in actual work
 
# TITLE PROGRESSION SCORING
 
def get_title_progression_score(career_history: list[dict]) -> tuple[float, dict]:
    """
    Score the candidate's seniority trajectory over time.
 
    SWE -> Senior SWE -> Staff Engineer -> Principal in 6 years = high score.
    Same title for 8 years = medium score.
    Already at Staff/Principal level = bonus.
    """
    SENIORITY_MAP = [
        ("intern", 0), ("trainee", 0),
        ("junior", 1), ("associate", 1), ("graduate", 1),
        ("senior", 3), ("specialist", 3),
        ("lead", 4), ("staff", 4),
        ("principal", 5), ("head", 5), ("architect", 5),
        ("director", 6), ("manager", 5),
        ("vp", 7), ("cto", 8), ("ceo", 8), ("co-founder", 7),
    ]
 
    def title_to_level(title: str) -> int:
        t = title.lower()
        level = 2  # default mid-level
        for keyword, lvl in SENIORITY_MAP:
            if keyword in t:
                level = max(level, lvl)
        return level
 
    if not career_history:
        return 0.5, {"reason": "no_history"}
 
    try:
        sorted_career = sorted(career_history, key=lambda x: x.get("start_date", "2000-01-01"))
    except Exception:
        sorted_career = career_history
 
    levels = [title_to_level(ch.get("title", "")) for ch in sorted_career]
    details = {"levels": levels}
 
    if len(levels) == 1:
        score = round(min(1.0, 0.4 + levels[0] * 0.08), 3)
        return score, details
 
    max_level   = max(levels)
    recent_max  = max(levels[-2:])
    early_max   = max(levels[:2])
 
    if recent_max > early_max:
        score = min(1.0, 0.60 + (recent_max - early_max) * 0.10)
    elif recent_max == early_max:
        score = 0.40 + recent_max * 0.08
    else:
        score = max(0.30, 0.50 - (early_max - recent_max) * 0.05)
 
    if max_level >= 5:
        score = min(1.0, score + 0.10)
 
    score = round(score, 3)
    details.update({"max_level": max_level, "recent_max": recent_max,
                    "early_max": early_max, "progression_score": score})
    return score, details

# HONEYPOT DETECTION
 
def is_honeypot(c: dict) -> bool:
    """
    Detect candidates with subtly impossible profiles.
    The submission spec warns: >10% honeypots in top 100 = disqualification.
    """
    # Signal 1: Expert/Advanced skill with 0 duration_months used
    expert_zero_duration = sum(
        1 for sk in c.get("skills", [])
        if sk.get("proficiency") in ("expert", "advanced")
        and sk.get("duration_months", 1) == 0
    )
    if expert_zero_duration >= 3:
        return True
 
    # Signal 2: Massive skill count ALL at expert/advanced with 0 endorsements each
    skills = c.get("skills", [])
    if len(skills) >= 8:
        high_with_zero = sum(
            1 for sk in skills
            if sk.get("proficiency") in ("expert", "advanced")
            and sk.get("endorsements", 0) == 0
            and sk.get("duration_months", 0) == 0
        )
        if high_with_zero >= 6:
            return True
 
    # Signal 3: Career timeline impossibility
    # (eg 8 years at company but duration_months is impossibly large)
    for ch in c.get("career_history", []):
        duration = ch.get("duration_months", 0)
        if duration > 600:  # 50+ years at one job = impossible
            return True
        try:
            start = datetime.strptime(ch.get("start_date", ""), "%Y-%m-%d").date()
            end_str = ch.get("end_date")
            if end_str:
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                claimed = ch.get("duration_months", 0)
                # More than 2 years of discrepancy = suspicious
                if claimed > 0 and abs(actual_months - claimed) > 24:
                    return True
        except (ValueError, TypeError):
            pass
 
    return False
 
# CAREER FIT SCORING (35% of final score — highest weight, by JD design)
 
def get_career_fit_score(c: dict) -> tuple[float, dict]:
    """
    Analyze career trajectory for:
    1. Product company vs consulting-only
    2. Right domain (AI/ML/search/retrieval)
    3. Production deployment evidence
    4. Seniority progression
    Returns score 0-1 and a breakdown dict for reasoning.
    """
    career = c.get("career_history", [])
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "").lower()
    details = {}
 
    if not career:
        return 0.1, {"reason": "no_career_history"}
 
    # Product vs consulting company analysis
    total_months = 0
    product_months = 0
    consulting_months = 0
    all_company_names = []
 
    for ch in career:
        co = ch.get("company", "").lower()
        dur = ch.get("duration_months", 0)
        total_months += dur
        all_company_names.append(co)
 
        is_consulting = any(cf in co for cf in CONSULTING_FIRM_PATTERNS)
        is_product = any(pf in co for pf in PRODUCT_COMPANY_SIGNALS)
 
        if is_consulting and not is_product:
            consulting_months += dur
        elif is_product:
            product_months += dur
 
    consulting_fraction = consulting_months / max(total_months, 1)
    product_fraction = product_months / max(total_months, 1)
    details["consulting_fraction"] = consulting_fraction
    details["product_fraction"] = product_fraction
 
    # Checking if ENTIRELY consulting
    entirely_consulting = all(
        any(cf in co for cf in CONSULTING_FIRM_PATTERNS)
        for co in all_company_names
    )
    details["entirely_consulting"] = entirely_consulting
 
    # Domain relevance: is this person actually in AI/ML/search?
    all_text = (
        " ".join(ch.get("description", "") for ch in career)
        + " " + profile.get("summary", "")
        + " " + profile.get("headline", "")
    ).lower()
    all_titles = " ".join(
        ch.get("title", "").lower() for ch in career
    ) + " " + current_title
 
    # AI/ML domain signals in career descriptions
    production_signals = [
        "deployed", "production", "shipped", "launched", "serving",
        "users", "scale", "latency", "throughput", "real-time",
        "a/b test", "experiment", "offline eval", "online eval",
        "retrieval", "ranking", "recommendation", "search", "embedding",
        "vector", "faiss", "elasticsearch", "hybrid", "rerank",
    ]
    production_hit_count = sum(1 for s in production_signals if s in all_text)
    details["production_signal_count"] = production_hit_count
 
    # Wrong domain: career entirely in unrelated field
    wrong_domain_hit = any(kw in all_titles for kw in WRONG_DOMAIN_TITLE_KEYWORDS)
    ai_title_hit = any(kw in all_titles for kw in AI_TITLE_KEYWORDS)
    # Also check generic "software engineer" / "backend" which the JD considers
    swe_hit = any(kw in all_titles for kw in {
        "software engineer", "backend", "full stack", "fullstack",
        "data engineer", "platform engineer", "infrastructure engineer",
    })
    details["ai_title_hit"] = ai_title_hit
    details["swe_hit"] = swe_hit
    details["wrong_domain"] = wrong_domain_hit
 
    # Years of experience vs JD preferred range (5-9, ideal 6-8)
    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        exp_score = 0.85
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        exp_score = 0.65
    elif yoe < 3:
        exp_score = 0.3
    else:
        exp_score = 0.5
    details["exp_score"] = exp_score
 
    # Title progression scoring
    progression_score, prog_details = get_title_progression_score(career)
    details["progression_score"] = progression_score
    details["progression_details"] = prog_details
 
    # Compose career score
    if wrong_domain_hit and not ai_title_hit and not swe_hit:
        domain_score = 0.05
    elif ai_title_hit:
        domain_score = 1.0
    elif swe_hit:
        domain_score = 0.7
    else:
        domain_score = 0.4
 
    if entirely_consulting:
        company_score = 0.2
    elif product_fraction > 0.5:
        company_score = 1.0
    elif product_fraction > 0.25:
        company_score = 0.75
    elif consulting_fraction < 0.5:
        company_score = 0.6
    else:
        company_score = 0.4
 
    if production_hit_count >= 8:
        prod_score = 1.0
    elif production_hit_count >= 5:
        prod_score = 0.8
    elif production_hit_count >= 2:
        prod_score = 0.6
    else:
        prod_score = 0.3
 
    # Updated weights: progression_score replaces some of domain + exp weight
    career_score = (
        domain_score      * 0.35   # was 0.40 — progression takes some weight
        + company_score   * 0.30   # was 0.35
        + prod_score      * 0.15
        + progression_score * 0.12  # NEW: seniority trajectory signal
        + exp_score       * 0.08   # was 0.10
    )
    details["domain_score"]   = domain_score
    details["company_score"]  = company_score
    details["prod_score"]     = prod_score
    details["career_score"]   = career_score
 
    return career_score, details
 
# SKILL FIT SCORING  (25% of final score)
 
def get_skill_fit_score(c: dict) -> tuple[float, dict]:
    """
    Score skill alignment with JD, validated by:
    - Endorsement count (social proof)
    - duration_months (actually used, not just listed)
    - Redrob skill assessment scores (platform-verified)
    - Proficiency level
 
    Critical: skills listed but never used (duration=0) or
    unendorsed are down-weighted. This catches keyword stuffers.
    """
    skills = c.get("skills", [])
    assessment_scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    details = {}
 
    if not skills:
        return 0.0, {"reason": "no_skills"}
 
    core_score = 0.0
    bonus_score = 0.0
    max_possible_core = len(CORE_SKILL_KEYWORDS)
 
    core_hits = []
    bonus_hits = []
 
    career_history = c.get("career_history", [])
 
    for sk in skills:
        name = sk.get("name", "").lower()
        prof = sk.get("proficiency", "beginner")
        endorsements = sk.get("endorsements", 0)
        duration = sk.get("duration_months", 0)
 
        # Proficiency weight
        prof_weight = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(prof, 0.3)
 
        # Trust multiplier: endorsed + used = real skill
        if duration == 0 and endorsements == 0:
            trust = 0.1  # keyword stuffing signal
        elif duration == 0:
            trust = 0.4
        elif endorsements == 0:
            trust = 0.6
        else:
            trust = min(1.0, 0.6 + endorsements / 100.0 + duration / 60.0 * 0.2)
 
        # Redrob assessment score
        assessment_key = next(
            (k for k in assessment_scores if k.lower() == name), None
        )
        if assessment_key:
            assessed = assessment_scores[assessment_key]
            assessment_factor = assessed / 100.0
            trust = (trust + assessment_factor) / 2.0
 
        # Cross-validate skill against career history
        # Halves trust for skills claimed but never evidenced in actual work
        cv_factor = cross_validate_skill(sk.get("name", ""), career_history)
        trust = trust * cv_factor
 
        effective_score = prof_weight * trust
 
        # Match against JD requirements
        is_core = any(keyword in name for keyword in CORE_SKILL_KEYWORDS)
        is_bonus = any(keyword in name for keyword in BONUS_SKILL_KEYWORDS)
 
        if is_core:
            core_score += effective_score
            core_hits.append(name)
        elif is_bonus:
            bonus_score += effective_score * 0.5
            bonus_hits.append(name)
 
    # Normalize: cap at 1.0
    # A candidate with 5 strong core skills (all expert, endorsed, used) gets ~1.0
    normalized_core = min(1.0, core_score / 5.0)
    normalized_bonus = min(0.3, bonus_score / 5.0)
    skill_score = min(1.0, normalized_core * 0.8 + normalized_bonus * 0.2)
 
    details["core_hits"] = core_hits[:5]
    details["bonus_hits"] = bonus_hits[:3]
    details["skill_score"] = skill_score
 
    return skill_score, details
 
# BEHAVIORAL SIGNAL SCORING  (15% of final score)
 
def get_behavioral_score(c: dict) -> tuple[float, dict]:
    """
    The JD explicitly states behavioral signals matter:
    "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is, for hiring purposes, not available."
    """
    sig = c.get("redrob_signals", {})
    profile = c.get("profile", {})
    details = {}
 
    # Recency: days since last active
    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (EVAL_DATE - last_active).days
    except (ValueError, TypeError):
        days_inactive = 365
 
    if days_inactive <= 14:
        recency_score = 1.0
    elif days_inactive <= 30:
        recency_score = 0.9
    elif days_inactive <= 60:
        recency_score = 0.75
    elif days_inactive <= 120:
        recency_score = 0.5
    elif days_inactive <= 180:
        recency_score = 0.25
    else:
        recency_score = 0.05  # 6+ months inactive: "not actually available"
    details["days_inactive"] = days_inactive
    details["recency_score"] = recency_score
 
    # Open-to-work flag (self-declared availability) 
    open_to_work = sig.get("open_to_work_flag", False)
    details["open_to_work"] = open_to_work
 
    # Recruiter response rate 
    response_rate = sig.get("recruiter_response_rate", 0)
    if response_rate >= 0.7:
        response_score = 1.0
    elif response_rate >= 0.4:
        response_score = 0.7
    elif response_rate >= 0.2:
        response_score = 0.4
    else:
        response_score = 0.1
    details["response_rate"] = response_rate
 
    # Notice period (JD: "we'd love sub-30-day notice")
    notice = sig.get("notice_period_days", 90)
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.3
    details["notice_days"] = notice
 
    # GitHub activity (JD mentions "open source contributions")
    github = sig.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.4  # neutral — no GitHub isn't necessarily bad
    elif github >= 70:
        github_score = 1.0
    elif github >= 40:
        github_score = 0.8
    elif github >= 15:
        github_score = 0.6
    else:
        github_score = 0.3
    details["github_score"] = github_score
 
    # Location / relocation (JD: Pune/Noida, Hyderabad, Mumbai, Delhi)
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    willing_relocate = sig.get("willing_to_relocate", False)
    preferred_mode = sig.get("preferred_work_mode", "flexible")
 
    in_target_city = any(city in location for city in TARGET_INDIA_LOCATIONS)
    in_india = country == "india"
 
    if in_target_city:
        location_score = 1.0
    elif in_india and willing_relocate:
        location_score = 0.85
    elif in_india:
        location_score = 0.7
    elif willing_relocate and country in ("", "unknown"):
        location_score = 0.5
    else:
        location_score = 0.3  # Outside India = possible but JD says case-by-case
    details["location_score"] = location_score
    details["location"] = f"{location}, {country}"
 
    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50)
    completeness_score = completeness / 100.0
 
    # Interview reliability signals
    interview_rate = sig.get("interview_completion_rate", 0.5)
    offer_rate = sig.get("offer_acceptance_rate", -1)
    reliability_score = interview_rate * 0.7
    if offer_rate >= 0:
        reliability_score = reliability_score * 0.5 + offer_rate * 0.5
 
    # Compose behavioral score
    behavioral_score = (
        recency_score * 0.30
        + (1.0 if open_to_work else 0.3) * 0.15
        + response_score * 0.20
        + notice_score * 0.10
        + github_score * 0.05
        + location_score * 0.10
        + completeness_score * 0.05
        + reliability_score * 0.05
    )
    details["behavioral_score"] = behavioral_score
    return behavioral_score, details
 
# REASONING GENERATOR  (for CSV reasoning column)
 
def generate_reasoning(c: dict, career_d: dict, skill_d: dict, behav_d: dict,
                       final_score: float, gemini_d: dict | None = None) -> str:
    """
    Generate 1-2 sentence specific, fact-grounded reasoning.
    Judges check: specific facts, JD connection, honest concerns, no hallucination.
    Incorporates Gemini insight when available (more varied, less templated).
    """
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    country = profile.get("country", "Unknown")
    core_hits = skill_d.get("core_hits", [])
    days_inactive = behav_d.get("days_inactive", 999)
    response_rate = behav_d.get("response_rate", 0)
    notice = behav_d.get("notice_days", 90)
    open_to_work = behav_d.get("open_to_work", False)
 
    strengths = []
    concerns = []
 
    # Strengths
    if career_d.get("ai_title_hit"):
        strengths.append(f"{yoe:.0f}-year career in AI/ML roles")
    elif career_d.get("swe_hit"):
        strengths.append(f"{yoe:.0f} years as {title}")
 
    if core_hits:
        skill_str = ", ".join(core_hits[:3])
        strengths.append(f"relevant skills including {skill_str}")
 
    if career_d.get("product_fraction", 0) > 0.4:
        strengths.append("meaningful product-company exposure")
 
    prod_count = career_d.get("production_signal_count", 0)
    if prod_count >= 6:
        strengths.append("strong production deployment evidence in career history")
 
    # Title progression signal
    prog = career_d.get("progression_details", {})
    max_level = prog.get("max_level", 0)
    prog_score = career_d.get("progression_score", 0.5)
    if max_level >= 5:
        strengths.append("reached Staff/Principal/Head level — strong seniority signal")
    elif prog_score >= 0.75:
        strengths.append("clear upward title progression across career")
 
    # Gemini insight 
    if gemini_d and gemini_d.get("reason"):
        strengths.append(f"AI assessment: {gemini_d['reason'][:80]}")
 
    if days_inactive <= 30:
        strengths.append("active on platform recently")
    if open_to_work:
        strengths.append("marked as open to work")
    if response_rate >= 0.7:
        strengths.append(f"high recruiter response rate ({response_rate:.0%})")
 
    # Concerns
    if career_d.get("entirely_consulting"):
        concerns.append("career appears consulting-firm-only (JD disqualifier)")
    if career_d.get("wrong_domain") and not career_d.get("ai_title_hit"):
        concerns.append("titles suggest non-technical domain background")
    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days (availability risk)")
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    if response_rate < 0.2:
        concerns.append(f"very low recruiter response rate ({response_rate:.0%})")
    if yoe < 4:
        concerns.append(f"below JD experience range ({yoe:.1f} years)")
 
    # 1-2 sentence reasoning
    sentence_1 = ""
    if strengths:
        sentence_1 = f"{title} ({yoe:.0f} yrs) with {'; '.join(strengths[:3])}."
    else:
        sentence_1 = f"{title} at {company} ({yoe:.0f} yrs experience, {country})."
 
    sentence_2 = ""
    if concerns:
        sentence_2 = f"Concerns: {'; '.join(concerns[:2])}."
    elif final_score >= 0.7:
        sentence_2 = "Strong match for JD core requirements with good engagement signals."
    elif final_score >= 0.5:
        sentence_2 = "Partial fit; some JD requirements met but gaps remain."
    else:
        sentence_2 = "Below cutoff — included as filler given experience level."
 
    return (sentence_1 + " " + sentence_2).strip()