"""
Synthetic Ticket Corpus for SupportTriageEnv.

All tickets are fully synthetic and deterministic. Ground-truth labels
(priority, department, and response keywords) are embedded directly, so
the grader never needs an external API.

The corpus is organised into three difficulty tiers:
- EASY  : clear signals, obvious priority + department
- MEDIUM: moderate ambiguity, multi-department overlap
- HARD  : complex, multi-issue, high-stakes, SLA pressure

Each ticket dict contains:
    id                  Unique string identifier
    text                Full ticket body (what the customer wrote)
    customer_tier       enterprise | pro | free
    previous_interactions  int >= 0
    is_repeat_complaint bool
    sla_steps           Steps before SLA breach (1 = very urgent)
    true_priority       ground-truth label
    true_department     ground-truth label
    response_keywords   list[str] — words a good response SHOULD contain
    response_avoid      list[str] — words a bad/generic response typically uses
    difficulty          easy | medium | hard
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Corpus data
# ---------------------------------------------------------------------------

TICKETS: List[Dict[str, Any]] = [
    # ────────────────────────────────────────────────────────────────────────
    # EASY TICKETS  (Task 1 — single ticket, 1 step)
    # ────────────────────────────────────────────────────────────────────────
    {
        "id": "T001",
        "text": (
            "Hi, I can't log in to my account. I've tried resetting my password "
            "three times and it's not working. I have a presentation in 2 hours "
            "and NEED access NOW. Please fix this immediately!"
        ),
        "customer_tier": "pro",
        "previous_interactions": 1,
        "is_repeat_complaint": False,
        "sla_steps": 1,
        "true_priority": "urgent",
        "true_department": "account",
        "response_keywords": ["password", "reset", "urgent", "access", "immediately", "assist"],
        "response_avoid": ["valued customer", "we apologize for the inconvenience", "soon as possible"],
        "difficulty": "easy",
    },
    {
        "id": "T002",
        "text": (
            "Hello, I'd like to update my billing address. Moving to a new office "
            "next month and want to make sure invoices go to the right place."
        ),
        "customer_tier": "free",
        "previous_interactions": 0,
        "is_repeat_complaint": False,
        "sla_steps": 5,
        "true_priority": "low",
        "true_department": "billing",
        "response_keywords": ["billing", "address", "update", "invoice", "settings"],
        "response_avoid": ["urgent", "immediately", "critical"],
        "difficulty": "easy",
    },
    {
        "id": "T003",
        "text": (
            "The entire API is down. None of our 50 developers can make requests. "
            "We're losing $10,000 per hour. This is a production outage!"
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 12,
        "is_repeat_complaint": False,
        "sla_steps": 1,
        "true_priority": "urgent",
        "true_department": "technical",
        "response_keywords": ["outage", "API", "production", "status", "team", "immediately", "escalate"],
        "response_avoid": ["normal business hours", "email your request"],
        "difficulty": "easy",
    },
    {
        "id": "T004",
        "text": (
            "Can you send me a copy of my invoice from last December? I need it "
            "for tax purposes."
        ),
        "customer_tier": "pro",
        "previous_interactions": 3,
        "is_repeat_complaint": False,
        "sla_steps": 8,
        "true_priority": "low",
        "true_department": "billing",
        "response_keywords": ["invoice", "December", "tax", "PDF", "send", "records"],
        "response_avoid": ["urgent", "critical", "outage"],
        "difficulty": "easy",
    },
    {
        "id": "T005",
        "text": (
            "I changed my email address but I'm no longer receiving notifications. "
            "Please help."
        ),
        "customer_tier": "free",
        "previous_interactions": 0,
        "is_repeat_complaint": False,
        "sla_steps": 6,
        "true_priority": "medium",
        "true_department": "account",
        "response_keywords": ["email", "notification", "settings", "verify", "update"],
        "response_avoid": ["critical", "outage", "immediately"],
        "difficulty": "easy",
    },
    {
        "id": "T006",
        "text": (
            "Our team's SSO integration is broken. Nobody can access the dashboard. "
            "We're an enterprise client and this is blocking 200 employees."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 8,
        "is_repeat_complaint": True,
        "sla_steps": 1,
        "true_priority": "urgent",
        "true_department": "technical",
        "response_keywords": ["SSO", "enterprise", "dashboard", "escalate", "immediately", "team"],
        "response_avoid": ["we will look into it", "please be patient"],
        "difficulty": "easy",
    },
    {
        "id": "T007",
        "text": (
            "I was accidentally charged twice for my subscription. "
            "Please refund the duplicate charge."
        ),
        "customer_tier": "pro",
        "previous_interactions": 2,
        "is_repeat_complaint": False,
        "sla_steps": 4,
        "true_priority": "high",
        "true_department": "billing",
        "response_keywords": ["charge", "refund", "duplicate", "subscription", "review", "process"],
        "response_avoid": ["cannot", "policy", "no refund"],
        "difficulty": "easy",
    },

    # ────────────────────────────────────────────────────────────────────────
    # MEDIUM TICKETS  (Task 2 — 5 tickets per episode)
    # ────────────────────────────────────────────────────────────────────────
    {
        "id": "T101",
        "text": (
            "Hi support, our data export has been running for 6 hours with no results. "
            "We have a compliance audit Monday and NEED this data. "
            "We're an enterprise subscriber."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 5,
        "is_repeat_complaint": False,
        "sla_steps": 2,
        "true_priority": "urgent",
        "true_department": "technical",
        "response_keywords": ["export", "audit", "compliance", "team", "escalate", "priority", "Monday"],
        "response_avoid": ["standard processing time", "wait"],
        "difficulty": "medium",
    },
    {
        "id": "T102",
        "text": (
            "I think my colleague accidentally deleted our shared workspace. "
            "Can we get it restored? Not super urgent but would like it back soon."
        ),
        "customer_tier": "pro",
        "previous_interactions": 1,
        "is_repeat_complaint": False,
        "sla_steps": 6,
        "true_priority": "medium",
        "true_department": "account",
        "response_keywords": ["workspace", "restore", "backup", "recovery", "team"],
        "response_avoid": ["cannot restore", "permanent", "deleted forever"],
        "difficulty": "medium",
    },
    {
        "id": "T103",
        "text": (
            "This is the THIRD time I'm writing about the same billing error. "
            "I keep getting charged for a plan I downgraded from 3 months ago. "
            "Your support has been completely useless. Fix this now."
        ),
        "customer_tier": "pro",
        "previous_interactions": 6,
        "is_repeat_complaint": True,
        "sla_steps": 2,
        "true_priority": "high",
        "true_department": "billing",
        "response_keywords": ["billing", "downgrade", "refund", "escalate", "resolve", "apologize", "review"],
        "response_avoid": ["as per policy", "cannot process", "review takes time"],
        "difficulty": "medium",
    },
    {
        "id": "T104",
        "text": (
            "We noticed some unusual API usage patterns in our logs that might indicate "
            "a security breach. Is there a way to audit API key usage? "
            "We're somewhat concerned but not sure if it's serious."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 15,
        "is_repeat_complaint": False,
        "sla_steps": 3,
        "true_priority": "high",
        "true_department": "technical",
        "response_keywords": ["security", "API", "audit", "key", "logs", "investigate", "team"],
        "response_avoid": ["no issues detected", "normal", "ignore"],
        "difficulty": "medium",
    },
    {
        "id": "T105",
        "text": (
            "How do I add a new team member to my account? I looked in settings "
            "but couldn't find the invite option."
        ),
        "customer_tier": "free",
        "previous_interactions": 0,
        "is_repeat_complaint": False,
        "sla_steps": 10,
        "true_priority": "low",
        "true_department": "account",
        "response_keywords": ["invite", "team", "member", "settings", "steps", "navigate"],
        "response_avoid": ["escalate", "critical", "urgent"],
        "difficulty": "medium",
    },
    {
        "id": "T106",
        "text": (
            "The webhook integration I set up yesterday is now firing duplicate events. "
            "We're receiving every event twice, which is causing duplicate records in our database."
        ),
        "customer_tier": "pro",
        "previous_interactions": 4,
        "is_repeat_complaint": False,
        "sla_steps": 3,
        "true_priority": "high",
        "true_department": "technical",
        "response_keywords": ["webhook", "duplicate", "event", "investigate", "configuration", "fix"],
        "response_avoid": ["normal behavior", "expected", "no issue"],
        "difficulty": "medium",
    },
    {
        "id": "T107",
        "text": (
            "I need to cancel my subscription effective immediately due to budget cuts. "
            "I also need a pro-rated refund for the remaining 18 days of my billing cycle."
        ),
        "customer_tier": "pro",
        "previous_interactions": 8,
        "is_repeat_complaint": False,
        "sla_steps": 5,
        "true_priority": "medium",
        "true_department": "billing",
        "response_keywords": ["cancel", "subscription", "refund", "pro-rated", "billing", "process"],
        "response_avoid": ["no refund", "non-refundable", "cannot cancel"],
        "difficulty": "medium",
    },

    # ────────────────────────────────────────────────────────────────────────
    # HARD TICKETS  (Task 3 — 3 tickets, full triage with response)
    # ────────────────────────────────────────────────────────────────────────
    {
        "id": "T201",
        "text": (
            "CRITICAL: Our production deployment pipeline just failed during a major "
            "release. The CI/CD system is throwing 'Authentication token expired' errors "
            "for all service accounts. We have 50,000 users waiting on this release. "
            "Our CEO is watching. This started exactly 23 minutes ago.\n\n"
            "Error log excerpt:\n"
            "ERROR [2026-03-15 14:32:11] ServiceAuth: Token validation failed for "
            "service-account-prod-deploy\n"
            "ERROR [2026-03-15 14:32:11] Pipeline halted: insufficient permissions\n\n"
            "We need immediate resolution. Our SLA with our clients is at risk."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 23,
        "is_repeat_complaint": False,
        "sla_steps": 1,
        "true_priority": "urgent",
        "true_department": "technical",
        "response_keywords": [
            "service account", "token", "expire", "pipeline", "escalate",
            "team", "immediately", "authenticate", "production", "priority"
        ],
        "response_avoid": [
            "business hours", "24-48 hours", "email us", "patience", "soon"
        ],
        "difficulty": "hard",
    },
    {
        "id": "T202",
        "text": (
            "I am writing on behalf of Meridian Analytics (enterprise account #ENT-8821). "
            "We have three separate issues that all need resolution:\n\n"
            "1. BILLING: We were invoiced $45,000 for Q1 but our contract cap is $38,000. "
            "Finance is asking us to escalate immediately.\n\n"
            "2. TECHNICAL: Our data pipeline integration broke after your last API update. "
            "We're missing 3 days of data that we need for a board presentation next week.\n\n"
            "3. ACCOUNT: We need to add 15 new users immediately as we've just closed Q2 planning.\n\n"
            "Please treat ALL of these as high priority. We've been a customer for 4 years."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 31,
        "is_repeat_complaint": True,
        "sla_steps": 2,
        "true_priority": "urgent",
        "true_department": "billing",
        "response_keywords": [
            "billing", "contract", "invoice", "escalate", "account manager",
            "pipeline", "data", "users", "team", "address each", "priority"
        ],
        "response_avoid": [
            "one at a time", "submit separate tickets", "standard process", "patience"
        ],
        "difficulty": "hard",
    },
    {
        "id": "T203",
        "text": (
            "My team upgraded our plan last week and now half our integrations are broken. "
            "The features that were supposed to be unlocked with the upgrade aren't showing. "
            "Meanwhile, we can't access some basic features we had before. "
            "I'm not sure if it's a billing issue or a technical issue — maybe both?\n\n"
            "Specific problems:\n"
            "- Advanced analytics tab shows 'Feature not available on your plan'\n"
            "- Slack integration that was working before now says 'Connection failed'\n"
            "- We're being charged the enterprise rate but seem to have pro features\n\n"
            "We have a demo with a client next Friday and need this working."
        ),
        "customer_tier": "enterprise",
        "previous_interactions": 7,
        "is_repeat_complaint": False,
        "sla_steps": 3,
        "true_priority": "high",
        "true_department": "technical",
        "response_keywords": [
            "upgrade", "plan", "integration", "analytics", "Slack", "billing",
            "review", "account", "investigate", "demo", "resolve"
        ],
        "response_avoid": [
            "submit separate tickets", "not our issue", "wait", "cannot help"
        ],
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_tickets_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Return all tickets matching the given difficulty tier."""
    return [t for t in TICKETS if t["difficulty"] == difficulty]


def get_ticket_by_id(ticket_id: str) -> Dict[str, Any]:
    """Return a ticket dict by its ID. Raises KeyError if not found."""
    for ticket in TICKETS:
        if ticket["id"] == ticket_id:
            return ticket
    raise KeyError(f"Ticket {ticket_id!r} not found in corpus")


# Priority adjacency map — adjacent levels get partial credit (0.5)
PRIORITY_ADJACENCY: Dict[str, List[str]] = {
    "urgent": ["high"],
    "high": ["urgent", "medium"],
    "medium": ["high", "low"],
    "low": ["medium"],
}

# Tier reward multipliers
TIER_MULTIPLIER: Dict[str, float] = {
    "enterprise": 2.0,
    "pro": 1.5,
    "free": 1.0,
}

# Department-priority combinations that are especially dangerous to mis-route
# (dept, priority) → extra_penalty when wrong
HIGH_RISK_COMBOS = {
    ("technical", "urgent"): 0.15,
    ("billing", "urgent"): 0.10,
    ("technical", "high"): 0.05,
}
