"""
SupportTriage Environment — Pydantic Models

Defines the typed Action, Observation, and State models used by the
SupportTriageEnv. These models extend openenv-core base types to
ensure full spec compliance.

Action space:
    - ticket_id: Which ticket the agent is triaging
    - priority: urgent / high / medium / low
    - department: billing / technical / account / general
    - response_draft: Agent-written first response (optional for Tasks 1 & 2)

Observation space:
    - ticket_id, ticket_text, summary metadata
    - customer_tier, previous_interactions, is_repeat_complaint
    - sla_deadline_steps: How many steps remain before SLA breach
    - remaining_tickets, step_reward, cumulative_reward

State:
    - episode_id, step_count (from base)
    - task_id, difficulty, tickets_processed, total_tickets
    - sla_breaches, compounding_mistake_count
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for local development without openenv-core installed
    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = Field(default=False)
        reward: Optional[float] = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# Priority and Department literals
# ---------------------------------------------------------------------------

PriorityLevel = Literal["urgent", "high", "medium", "low"]
Department = Literal["billing", "technical", "account", "general"]
CustomerTier = Literal["enterprise", "pro", "free"]
TaskDifficulty = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class SupportTriageAction(Action):
    """
    Agent action for triaging a single support ticket.

    The agent must classify priority and route to a department.
    For Task 3 (hard), a response_draft is also expected.

    Fields:
        ticket_id       ID of the ticket being triaged (must match current ticket)
        priority        Urgency classification: urgent > high > medium > low
        department      Target team: billing | technical | account | general
        response_draft  Written response to customer (required for Task 3)
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ticket_id: str = Field(
        ...,
        description="ID of the ticket being triaged. Must match the current ticket.",
        min_length=1,
        max_length=50,
    )
    priority: PriorityLevel = Field(
        ...,
        description="Urgency level: 'urgent' (SLA < 1h), 'high' (< 4h), "
                    "'medium' (< 24h), 'low' (< 72h)",
    )
    department: Department = Field(
        ...,
        description=(
            "Routing target: 'billing' (payments/invoices), "
            "'technical' (bugs/outages), 'account' (access/settings), "
            "'general' (misc)"
        ),
    )
    response_draft: str = Field(
        default="",
        description=(
            "Agent's draft response to the customer. "
            "Required and scored for Task 3 (hard). "
            "Ideal length: 50–150 words."
        ),
        max_length=2000,
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SupportTriageObservation(Observation):
    """
    Environment observation returned after each reset() or step().

    Contains the current ticket to triage plus contextual metadata
    about the customer and episode progress.

    Fields:
        ticket_id               Unique identifier for this ticket
        ticket_text             Full text of the support ticket
        customer_tier           Account tier: enterprise | pro | free
        previous_interactions   # prior tickets from this customer (indicates loyalty/frustration)
        is_repeat_complaint     True if this is a repeated issue from the same customer
        sla_deadline_steps      Steps remaining before SLA breach (0 = already breached)
        remaining_tickets       How many tickets remain in this episode (including current)
        last_action_feedback    Feedback dict from previous step (None on first step)
        step_reward             Reward from the most recent step (None on reset)
        cumulative_reward       Sum of rewards so far this episode
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Current ticket details
    ticket_id: str = Field(..., description="Unique ticket identifier")
    ticket_text: str = Field(..., description="Full text of the support ticket")

    # Customer context (affects reward weights)
    customer_tier: CustomerTier = Field(
        ...,
        description="Customer tier: 'enterprise' (2× reward weight), "
                    "'pro' (1.5×), 'free' (1×)",
    )
    previous_interactions: int = Field(
        default=0,
        ge=0,
        description="Number of prior tickets/chats from this customer",
    )
    is_repeat_complaint: bool = Field(
        default=False,
        description="True if same complaint category was filed before",
    )

    # SLA / time pressure
    sla_deadline_steps: int = Field(
        default=3,
        ge=0,
        description=(
            "Steps remaining before SLA breach. "
            "0 = already breached at this step. "
            "Breached tickets trigger an automatic penalty."
        ),
    )

    # Episode progress
    remaining_tickets: int = Field(
        default=1,
        ge=0,
        description="Number of tickets left in the queue including the current one",
    )

    # Feedback from previous step (None on reset)
    last_action_feedback: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Grading breakdown from the previous step: "
            "{'priority_score', 'routing_score', 'response_score', "
            "'sla_penalty', 'tier_multiplier', 'total_reward', 'notes'}"
        ),
    )

    # Reward signals
    step_reward: Optional[float] = Field(
        default=None,
        description="Reward earned in the most recent step",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated this episode",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SupportTriageState(State):
    """
    Full internal episode state (returned by state() endpoint).

    Extends the OpenEnv base State with task-specific fields.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Task metadata
    task_id: str = Field(default="task_1_easy", description="Current task identifier")
    difficulty: TaskDifficulty = Field(
        default="easy", description="Task difficulty: easy | medium | hard"
    )

    # Progress tracking
    tickets_processed: int = Field(
        default=0, ge=0, description="Tickets triaged so far this episode"
    )
    total_tickets: int = Field(
        default=1, ge=1, description="Total tickets in this episode"
    )

    # Penalty tracking
    sla_breaches: int = Field(
        default=0, ge=0, description="Number of SLA deadlines missed this episode"
    )
    compounding_mistake_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of mis-classifications in a row. "
            "Each additional mistake adds a compounding penalty."
        ),
    )
    consecutive_wrong: int = Field(
        default=0,
        ge=0,
        description="Consecutive wrong answers (resets to 0 on correct answer)",
    )

    # Cumulative stats
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated this episode"
    )
    max_possible_reward: float = Field(
        default=1.0, description="Maximum achievable reward for normalisation"
    )
