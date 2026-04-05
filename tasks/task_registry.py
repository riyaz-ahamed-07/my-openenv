"""
Task Registry for SupportTriageEnv.

Defines the three tasks (easy → medium → hard) and handles
episode construction (which tickets go into each episode).

Task definitions:
    task_1_easy    1 ticket, priority classification only
    task_2_medium  5 tickets, priority + routing
    task_3_hard    3 tickets, full triage (priority + routing + response)

Each task dict contains:
    id, name, difficulty, description
    ticket_ids       ordered list of ticket IDs for this episode
    max_steps        maximum agent steps before forced termination
    weight_priority  weight in reward computation
    weight_routing   weight in reward computation
    weight_response  weight in reward computation
    grade_response   bool — whether response_draft is graded
    sla_enabled      bool — whether SLA pressure is active
"""

import random
from typing import Any, Dict, List, Optional

from tasks.ticket_corpus import get_ticket_by_id, get_tickets_by_difficulty


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    # ────────────────────────────────────────────────────────────── EASY ──
    "task_1_easy": {
        "id": "task_1_easy",
        "name": "Single-Ticket Priority Classification",
        "difficulty": "easy",
        "description": (
            "Given a single support ticket, classify its priority level (urgent / high / medium / low). "
            "The agent must read the ticket carefully and identify urgency cues. "
            "No routing or response drafting required."
        ),
        "ticket_ids": ["T001", "T002", "T003", "T004", "T005", "T006", "T007"],  # pool
        "episode_ticket_ids": ["T003"],   # default deterministic episode
        "max_steps": 1,
        "weight_priority": 1.0,
        "weight_routing": 0.0,
        "weight_response": 0.0,
        "grade_response": False,
        "sla_enabled": True,
    },

    # ─────────────────────────────────────────────────────────── MEDIUM ──
    "task_2_medium": {
        "id": "task_2_medium",
        "name": "Multi-Ticket Routing and Priority",
        "difficulty": "medium",
        "description": (
            "Given a workload of 5 support tickets, classify the priority AND route each ticket "
            "to the correct department (billing / technical / account / general). "
            "Customer tier and SLA deadlines create time pressure — handle urgent tickets first. "
            "Reward is earned per-ticket and updates after every step."
        ),
        "ticket_ids": ["T101", "T102", "T103", "T104", "T105", "T106", "T107"],
        "episode_ticket_ids": ["T101", "T102", "T103", "T104", "T105"],  # default episode
        "max_steps": 5,
        "weight_priority": 0.5,
        "weight_routing": 0.5,
        "weight_response": 0.0,
        "grade_response": False,
        "sla_enabled": True,
    },

    # ──────────────────────────────────────────────────────────── HARD ──
    "task_3_hard": {
        "id": "task_3_hard",
        "name": "Full Triage Pipeline with Response Drafting",
        "difficulty": "hard",
        "description": (
            "Given 3 complex, high-stakes support tickets, perform complete triage: "
            "classify priority, route to the correct department, AND draft a professional "
            "initial response to the customer. "
            "Response quality is scored on a rubric: acknowledgement, action steps, "
            "department context, tone, and conciseness. "
            "All tickets are from enterprise customers with strict SLAs. "
            "Mistakes compound — consecutive errors incur escalating penalties."
        ),
        "ticket_ids": ["T201", "T202", "T203"],
        "episode_ticket_ids": ["T201", "T202", "T203"],
        "max_steps": 3,
        "weight_priority": 0.25,
        "weight_routing": 0.25,
        "weight_response": 0.50,
        "grade_response": True,
        "sla_enabled": True,
    },
}


# ---------------------------------------------------------------------------
# Episode builder
# ---------------------------------------------------------------------------


class Episode:
    """
    Represents a single episode (a sequence of tickets to triage).

    Provides an iterator interface: each call to next_ticket() advances
    the queue and returns (ticket_dict, sla_already_breached).
    """

    def __init__(self, task_id: str, seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASKS.keys())}"
            )
        self.task = TASKS[task_id]
        self.seed = seed

        # Build the ordered ticket list for this episode
        self._ticket_ids: List[str] = list(self.task["episode_ticket_ids"])
        self._tickets: List[Dict[str, Any]] = [
            get_ticket_by_id(tid) for tid in self._ticket_ids
        ]

        # SLA step counters: how many *agent steps* remain before SLA breach
        # These count down as steps pass.
        self._sla_counters: Dict[str, int] = {
            t["id"]: t["sla_steps"] for t in self._tickets
        }

        self._current_index: int = 0
        self._steps_taken: int = 0
        self.done: bool = False

    @property
    def total_tickets(self) -> int:
        return len(self._tickets)

    @property
    def remaining_tickets(self) -> int:
        return max(0, self.total_tickets - self._current_index)

    @property
    def current_ticket(self) -> Optional[Dict[str, Any]]:
        if self._current_index < len(self._tickets):
            return self._tickets[self._current_index]
        return None

    @property
    def current_sla_steps(self) -> int:
        """Steps remaining before current ticket's SLA is breached."""
        ticket = self.current_ticket
        if ticket is None:
            return 0
        return max(0, self._sla_counters[ticket["id"]])

    @property
    def current_sla_breached(self) -> bool:
        """True if SLA has already expired for the current ticket."""
        return self.current_sla_steps <= 0

    def advance(self) -> None:
        """Move to the next ticket in queue. Decrements all SLA counters."""
        # Decrement SLA for all remaining tickets
        for tid in self._sla_counters:
            self._sla_counters[tid] = max(0, self._sla_counters[tid] - 1)

        self._current_index += 1
        self._steps_taken += 1

        if self._current_index >= len(self._tickets):
            self.done = True


def get_task(task_id: str) -> Dict[str, Any]:
    """Return task config dict."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id!r}. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[Dict[str, str]]:
    """Return a summary list of all tasks (id, name, difficulty)."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
        }
        for t in TASKS.values()
    ]
