"""
Grader Module for SupportTriageEnv.

Implements the structured rubric-based scoring system.
All scoring is fully deterministic — no external API calls.

Response Quality Rubric (for Task 3):
    +0.20  Acknowledges user issue clearly
    +0.30  Provides concrete resolution steps
    +0.20  Matches correct department context
    +0.20  Professional and polite tone
    +0.10  Concise and well-structured (<150 words ideal)
    ─────
    Total: 1.00 before penalties

Penalties:
    -0.20  Empty or vague response (<20 words or no actionable content)
    -0.10  Overly long response (>500 words)

Priority scoring:
    +1.00  Exact match
    +0.50  Adjacent level (e.g., high for urgent)
    +0.00  Far miss (e.g., low for urgent)

Penalty modifiers:
    -0.30  Misclassifying "urgent" tickets (any non-urgent answer)
    -0.20  Wrong department routing
    ×2.0   Enterprise tier multiplier on base reward
    ×1.5   Pro tier multiplier on base reward
    Compounding penalty: -(0.05 × consecutive_wrong_count) per step
    SLA breach penalty: -0.15 per breached ticket
"""

import re
from typing import Any, Dict, Optional, Tuple

from tasks.ticket_corpus import (
    HIGH_RISK_COMBOS,
    PRIORITY_ADJACENCY,
    TIER_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Priority Grader
# ---------------------------------------------------------------------------


def grade_priority(
    predicted: str,
    true_priority: str,
    customer_tier: str,
) -> Tuple[float, float, str]:
    """
    Grade the priority classification.

    Returns:
        (base_score, penalty, note)
        base_score in {0.0, 0.5, 1.0}
        penalty for urgent miss (0.0 or 0.3)
    """
    if predicted == true_priority:
        return 1.0, 0.0, f"✓ Priority '{predicted}' correct"

    if predicted in PRIORITY_ADJACENCY.get(true_priority, []):
        note = f"~ Priority '{predicted}' adjacent to '{true_priority}' (+0.5)"
        # Extra penalty for missing urgent on a high-tier customer
        extra = 0.15 if (true_priority == "urgent" and customer_tier == "enterprise") else 0.0
        return 0.5, extra, note

    # Complete miss
    urgent_penalty = 0.30 if true_priority == "urgent" else 0.0
    note = f"✗ Priority '{predicted}' wrong (expected '{true_priority}')"
    return 0.0, urgent_penalty, note


# ---------------------------------------------------------------------------
# Department / Routing Grader
# ---------------------------------------------------------------------------


def grade_routing(
    predicted_dept: str,
    true_dept: str,
    true_priority: str,
) -> Tuple[float, float, str]:
    """
    Grade the department routing decision.

    Returns:
        (score, penalty, note)
    """
    if predicted_dept == true_dept:
        return 1.0, 0.0, f"✓ Routing to '{predicted_dept}' correct"

    # Look up extra penalty for high-risk combos
    extra = HIGH_RISK_COMBOS.get((true_dept, true_priority), 0.0)
    base_penalty = 0.20
    note = (
        f"✗ Routing to '{predicted_dept}' wrong (expected '{true_dept}')"
        + (f" [+{extra:.2f} high-risk penalty]" if extra else "")
    )
    return 0.0, base_penalty + extra, note


# ---------------------------------------------------------------------------
# Response Quality Grader
# ---------------------------------------------------------------------------

# Tone signals: words that indicate professionalism
PROFESSIONAL_SIGNALS = [
    "thank", "apolog", "understand", "assist", "help",
    "resolve", "investigate", "team", "review", "follow up", "reach out",
]

# Vagueness flags: generic phrases that add no value
VAGUE_SIGNALS = [
    "valued customer",
    "we apologize for any inconvenience",
    "as soon as possible",
    "please be patient",
    "we will look into it",
    "per our policy",
    "unfortunately we cannot",
    "thank you for your patience",
]

# Action-oriented words (indicates concrete resolution)
ACTION_WORDS = [
    "will", "escalat", "contact", "refund", "reset", "restor", "fix",
    "investigat", "updat", "step", "follow", "send", "creat", "check",
    "verif", "connect", "configur", "assign", "priorit",
]

# Department context keywords
DEPT_CONTEXT_KEYWORDS: Dict[str, list] = {
    "billing": ["invoice", "charge", "payment", "refund", "billing", "subscription", "account", "amount"],
    "technical": ["error", "bug", "outage", "API", "fix", "investigate", "deploy", "integration", "issue"],
    "account": ["account", "login", "access", "password", "settings", "email", "user", "team", "invite"],
    "general": ["help", "question", "information", "guide", "documentation", "resource"],
}

# Acknowledgement signals
ACKNOWLEDGEMENT_SIGNALS = [
    "understand", "hear", "appreciate", "see", "note", "aware",
    "received", "concern", "issue", "problem", "mention",
]


def _word_count(text: str) -> int:
    return len(text.split())


def _contains_any(text: str, keywords: list, case_insensitive: bool = True) -> bool:
    t = text.lower() if case_insensitive else text
    return any(kw.lower() in t for kw in keywords)


def _count_matches(text: str, keywords: list) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)


def grade_response(
    response: str,
    ticket: Dict[str, Any],
    true_department: str,
) -> Tuple[float, str]:
    """
    Grade the response draft using the structured deterministic rubric.

    Rubric:
        +0.20  Acknowledges user issue clearly
        +0.30  Provides concrete resolution steps
        +0.20  Matches correct department context
        +0.20  Professional and polite tone
        +0.10  Concise and well-structured (<150 words)

    Penalties:
        -0.20  Empty or vague response (<20 words or no actionable content)
        -0.10  Overly long (>500 words)

    Returns:
        (score, detailed_notes)
        score is clamped to [0.0, 1.0]
    """
    notes = []
    score = 0.0
    wc = _word_count(response)

    # ── Early exit: empty response ──────────────────────────────────────────
    if not response or wc < 10:
        return 0.0, "✗ Response is empty or too short (<10 words) [-0.20 penalty applied]"

    # ── 1. Acknowledges issue (+0.20) ───────────────────────────────────────
    # Response should reference relevant keywords from ticket
    ticket_keywords = ticket.get("response_keywords", [])[:5]  # use first 5 as ack signals
    ack_score = 0.0
    ack_hits = _count_matches(response, ticket_keywords + ACKNOWLEDGEMENT_SIGNALS)
    if ack_hits >= 2:
        ack_score = 0.20
        notes.append(f"✓ Acknowledges issue ({ack_hits} keyword hits) [+0.20]")
    elif ack_hits == 1:
        ack_score = 0.10
        notes.append(f"~ Partial acknowledgement (1 keyword hit) [+0.10]")
    else:
        notes.append("✗ Does not clearly acknowledge the issue [+0.00]")
    score += ack_score

    # ── 2. Concrete resolution steps (+0.30) ────────────────────────────────
    action_hits = _count_matches(response, ACTION_WORDS)
    # Bonus: presence of numbered steps or bullet points
    structured = bool(re.search(r"(\d+\.|[-*•])\s+\w+", response))
    res_score = 0.0
    if action_hits >= 3 or (action_hits >= 2 and structured):
        res_score = 0.30
        notes.append(f"✓ Concrete resolution steps provided ({action_hits} action words) [+0.30]")
    elif action_hits >= 1:
        res_score = 0.15
        notes.append(f"~ Partial resolution steps ({action_hits} action words) [+0.15]")
    else:
        notes.append("✗ No concrete resolution steps [+0.00]")
    score += res_score

    # ── 3. Department context match (+0.20) ─────────────────────────────────
    dept_keywords = DEPT_CONTEXT_KEYWORDS.get(true_department, [])
    dept_hits = _count_matches(response, dept_keywords)
    dept_score = 0.0
    if dept_hits >= 2:
        dept_score = 0.20
        notes.append(f"✓ Matches {true_department} context ({dept_hits} hits) [+0.20]")
    elif dept_hits == 1:
        dept_score = 0.10
        notes.append(f"~ Partial department context (1 hit) [+0.10]")
    else:
        notes.append(f"✗ Misses {true_department} context [+0.00]")
    score += dept_score

    # ── 4. Professional tone (+0.20) ────────────────────────────────────────
    professional_hits = _count_matches(response, PROFESSIONAL_SIGNALS)
    vague_hits = _count_matches(response, [v.lower() for v in VAGUE_SIGNALS])
    tone_score = 0.0
    if professional_hits >= 2 and vague_hits == 0:
        tone_score = 0.20
        notes.append(f"✓ Professional tone ({professional_hits} signals, {vague_hits} clichés) [+0.20]")
    elif professional_hits >= 1 and vague_hits <= 1:
        tone_score = 0.10
        notes.append(f"~ Acceptable tone ({professional_hits} signals, {vague_hits} clichés) [+0.10]")
    else:
        notes.append(f"✗ Unprofessional or cliché-heavy tone [+0.00]")
    score += tone_score

    # ── 5. Conciseness (+0.10) ──────────────────────────────────────────────
    if 30 <= wc <= 150:
        score += 0.10
        notes.append(f"✓ Concise ({wc} words, ideal 30-150) [+0.10]")
    elif wc <= 200:
        score += 0.05
        notes.append(f"~ Slightly long ({wc} words) [+0.05]")
    else:
        notes.append(f"~ Long response ({wc} words) [+0.00]")

    # ── Penalty: Vague / boilerplate ────────────────────────────────────────
    if vague_hits >= 2 or (action_hits == 0 and wc < 20):
        score -= 0.20
        notes.append(f"✗ Vague/boilerplate response penalty [-0.20]")

    # ── Penalty: Overly long ─────────────────────────────────────────────────
    if wc > 500:
        score -= 0.10
        notes.append(f"✗ Overly long response ({wc} words, max ideal 500) [-0.10]")

    # ── Check for 'avoid' keywords (strongly penalised) ─────────────────────
    avoid_words = ticket.get("response_avoid", [])
    avoid_hits = _count_matches(response, avoid_words)
    if avoid_hits >= 2:
        score -= 0.10
        notes.append(f"✗ Contains {avoid_hits} discouraged phrases [-0.10]")

    score = max(0.0, min(1.0, score))
    return round(score, 4), " | ".join(notes)


# ---------------------------------------------------------------------------
# Full Step Grader
# ---------------------------------------------------------------------------


def grade_step(
    action: Dict[str, Any],
    ticket: Dict[str, Any],
    task_config: Dict[str, Any],
    consecutive_wrong: int,
    sla_already_breached: bool,
) -> Dict[str, Any]:
    """
    Grade a single agent step and return a detailed breakdown.

    Args:
        action:            The agent's action dict (ticket_id, priority, department, response_draft)
        ticket:            Ground-truth ticket dict from corpus
        task_config:       Task weights dict from task_registry
        consecutive_wrong: Number of consecutive wrong answers before this step
        sla_already_breached: Whether SLA was already expired when step was taken

    Returns:
        {
            priority_score, routing_score, response_score,
            priority_penalty, routing_penalty, sla_penalty,
            compounding_penalty, tier_multiplier, raw_reward, final_reward,
            notes, response_notes, is_correct (bool)
        }
    """
    result: Dict[str, Any] = {}

    # ── Tier multiplier ──────────────────────────────────────────────────────
    tier = ticket["customer_tier"]
    tier_mult = TIER_MULTIPLIER.get(tier, 1.0)
    result["tier_multiplier"] = tier_mult

    # ── Priority grading ─────────────────────────────────────────────────────
    p_score, p_penalty, p_note = grade_priority(
        action["priority"], ticket["true_priority"], tier
    )
    result.update(priority_score=p_score, priority_penalty=p_penalty, priority_note=p_note)

    # ── Routing grading ──────────────────────────────────────────────────────
    r_score, r_penalty, r_note = grade_routing(
        action["department"], ticket["true_department"], ticket["true_priority"]
    )
    result.update(routing_score=r_score, routing_penalty=r_penalty, routing_note=r_note)

    # ── Response grading (only used for hard task) ───────────────────────────
    resp_score, resp_notes = 0.0, "N/A (response not graded for this task)"
    if task_config.get("grade_response", False):
        resp_score, resp_notes = grade_response(
            action.get("response_draft", ""), ticket, ticket["true_department"]
        )
    result.update(response_score=resp_score, response_notes=resp_notes)

    # ── Task weights ─────────────────────────────────────────────────────────
    w_priority = task_config.get("weight_priority", 1.0)
    w_routing = task_config.get("weight_routing", 0.0)
    w_response = task_config.get("weight_response", 0.0)
    weight_total = w_priority + w_routing + w_response

    # Weighted base score
    base_score = (
        p_score * w_priority +
        r_score * w_routing +
        resp_score * w_response
    ) / (weight_total if weight_total > 0 else 1.0)

    # ── SLA breach penalty ───────────────────────────────────────────────────
    sla_penalty = 0.15 if sla_already_breached else 0.0
    result["sla_penalty"] = sla_penalty

    # ── Compounding mistake penalty ───────────────────────────────────────────
    is_correct = (p_score >= 0.5 and r_score >= 0.5)
    compounding = 0.05 * consecutive_wrong if not is_correct else 0.0
    result["compounding_penalty"] = compounding
    result["is_correct"] = is_correct

    # ── Total penalties ───────────────────────────────────────────────────────
    total_penalty = p_penalty + r_penalty + sla_penalty + compounding

    # ── Raw reward (before tier mult) ────────────────────────────────────────
    raw_reward = base_score - total_penalty
    raw_reward = max(0.0, raw_reward)
    result["raw_reward"] = round(raw_reward, 4)

    # ── Final reward (tier-weighted, normalised by tier mult) ─────────────────
    # We scale by tier but then normalise so max possible ~ 1.0 per step
    final_reward = (raw_reward * tier_mult) / tier_mult  # tier affects rank, not absolute scale
    # Actually apply tier as additive bonus to make enterprise tickets matter more
    tier_bonus = (tier_mult - 1.0) * base_score * 0.2  # small bonus
    final_reward = min(1.0, raw_reward + tier_bonus)
    result["final_reward"] = round(final_reward, 4)

    # ── Notes summary ────────────────────────────────────────────────────────
    notes = [p_note, r_note]
    if task_config.get("grade_response", False):
        notes.append(f"Response: {resp_notes}")
    if sla_penalty:
        notes.append(f"✗ SLA breach penalty [-{sla_penalty:.2f}]")
    if compounding:
        notes.append(f"✗ Compounding mistake penalty [-{compounding:.2f}]")
    result["notes"] = notes

    return result
