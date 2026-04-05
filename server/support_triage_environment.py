"""
SupportTriageEnvironment — Server-Side Logic.

Implements the OpenEnv Environment interface for the customer support
triage task. This class runs inside the Docker container / HF Space.

Lifecycle:
    env = SupportTriageEnvironment()
    obs = env.reset(task_id="task_2_medium")   # start a new episode
    obs = env.step(action)                      # triage one ticket
    ...
    state = env.state                           # inspect episode state
"""

import uuid
from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server.environment import Environment
except ImportError:
    # Fallback base classes for local dev / testing without docker
    from pydantic import BaseModel, ConfigDict, Field

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

    class Environment:  # type: ignore[no-redef]
        pass

# Import our models
import sys
import os
# Ensure the parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SupportTriageAction, SupportTriageObservation, SupportTriageState
from tasks.task_registry import Episode, TASKS, list_tasks
from tasks.grader import grade_step


class SupportTriageEnvironment(Environment):
    """
    Server-side environment for Customer Support Triage.

    Manages episode state, ticket queuing, SLA tracking, and reward computation.
    Each WebSocket session gets its own instance (via class factory in create_app).

    Attributes:
        _episode     Current Episode instance (ticket queue + SLA counters)
        _state       SupportTriageState (tracks step count, penalties, rewards)
        _task_id     Active task identifier
    """

    def __init__(self) -> None:
        self._episode: Optional[Episode] = None
        self._state: SupportTriageState = SupportTriageState(
            episode_id=None,
            step_count=0,
            task_id="task_1_easy",
            difficulty="easy",
            tickets_processed=0,
            total_tickets=1,
            sla_breaches=0,
            compounding_mistake_count=0,
            consecutive_wrong=0,
            cumulative_reward=0.0,
            max_possible_reward=1.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """
        Start a new episode.

        Args:
            seed:       Random seed (unused — corpus is deterministic)
            task_id:    Which task to run: task_1_easy | task_2_medium | task_3_hard
                        Defaults to task_1_easy.
            episode_id: Custom episode identifier (auto-generated if None)

        Returns:
            Initial observation with the first ticket to triage.
        """
        resolved_task_id = task_id or kwargs.get("task_id", "task_1_easy")

        # Validate task_id
        if resolved_task_id not in TASKS:
            resolved_task_id = "task_1_easy"

        task_cfg = TASKS[resolved_task_id]

        # Create fresh episode
        self._episode = Episode(task_id=resolved_task_id, seed=seed)

        # Reset state
        self._state = SupportTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=resolved_task_id,
            difficulty=task_cfg["difficulty"],
            tickets_processed=0,
            total_tickets=self._episode.total_tickets,
            sla_breaches=0,
            compounding_mistake_count=0,
            consecutive_wrong=0,
            cumulative_reward=0.0,
            max_possible_reward=float(self._episode.total_tickets),
        )

        return self._build_observation(
            step_reward=None,
            last_feedback=None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """
        Execute one triage action for the current ticket.

        Args:
            action: SupportTriageAction with priority, department, response_draft

        Returns:
            Observation with reward, feedback, and next ticket (or done=True).
        """
        if self._episode is None or self._episode.done:
            # No active episode — return terminal observation
            return SupportTriageObservation(
                ticket_id="none",
                ticket_text="No active episode. Call reset() first.",
                customer_tier="free",
                previous_interactions=0,
                is_repeat_complaint=False,
                sla_deadline_steps=0,
                remaining_tickets=0,
                last_action_feedback=None,
                step_reward=0.0,
                cumulative_reward=self._state.cumulative_reward,
                done=True,
                reward=0.0,
                metadata={"error": "No active episode. Call reset() first."},
            )

        ticket = self._episode.current_ticket
        task_cfg = TASKS[self._state.task_id]

        # Convert action to dict for grader
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif hasattr(action, "dict"):
            action_dict = action.dict()
        else:
            action_dict = dict(action)

        # Check SLA
        sla_breached = self._episode.current_sla_breached

        # Grade the action
        grade = grade_step(
            action=action_dict,
            ticket=ticket,
            task_config=task_cfg,
            consecutive_wrong=self._state.consecutive_wrong,
            sla_already_breached=sla_breached,
        )

        step_reward = grade["final_reward"]

        # Update state
        self._state.step_count += 1
        self._state.tickets_processed += 1
        self._state.cumulative_reward += step_reward

        if sla_breached:
            self._state.sla_breaches += 1

        if grade["is_correct"]:
            self._state.consecutive_wrong = 0
        else:
            self._state.consecutive_wrong += 1
            self._state.compounding_mistake_count += 1

        # Advance episode queue
        self._episode.advance()

        # Build feedback for next observation
        feedback = {
            "priority_score": grade["priority_score"],
            "routing_score": grade["routing_score"],
            "response_score": grade["response_score"],
            "sla_penalty": grade["sla_penalty"],
            "compounding_penalty": grade["compounding_penalty"],
            "tier_multiplier": grade["tier_multiplier"],
            "total_reward": grade["final_reward"],
            "notes": grade["notes"],
        }
        if task_cfg.get("grade_response", False):
            feedback["response_notes"] = grade.get("response_notes", "")

        done = self._episode.done

        return self._build_observation(
            step_reward=step_reward,
            last_feedback=feedback,
            done=done,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Async variant (used by WebSocket handler)
    # ─────────────────────────────────────────────────────────────────────────

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """Async wrapper for step() (WebSocket handler calls this)."""
        return self.step(action, timeout_s=timeout_s, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # state property
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> SupportTriageState:
        """Return the current episode state."""
        return self._state

    # ─────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        step_reward: Optional[float],
        last_feedback: Optional[Dict[str, Any]],
        done: bool = False,
    ) -> SupportTriageObservation:
        """Construct an observation from the current episode state."""
        if self._episode is None or self._episode.current_ticket is None:
            # Terminal observation
            return SupportTriageObservation(
                ticket_id="done",
                ticket_text="All tickets have been processed. Episode complete.",
                customer_tier="free",
                previous_interactions=0,
                is_repeat_complaint=False,
                sla_deadline_steps=0,
                remaining_tickets=0,
                last_action_feedback=last_feedback,
                step_reward=step_reward,
                cumulative_reward=self._state.cumulative_reward,
                done=True,
                reward=step_reward,
                metadata={
                    "episode_summary": {
                        "total_reward": self._state.cumulative_reward,
                        "max_possible": self._state.max_possible_reward,
                        "normalized_score": round(
                            self._state.cumulative_reward / max(1, self._state.max_possible_reward), 4
                        ),
                        "sla_breaches": self._state.sla_breaches,
                        "tickets_processed": self._state.tickets_processed,
                    }
                },
            )

        ticket = self._episode.current_ticket

        return SupportTriageObservation(
            ticket_id=ticket["id"],
            ticket_text=ticket["text"],
            customer_tier=ticket["customer_tier"],
            previous_interactions=ticket["previous_interactions"],
            is_repeat_complaint=ticket["is_repeat_complaint"],
            sla_deadline_steps=self._episode.current_sla_steps,
            remaining_tickets=self._episode.remaining_tickets,
            last_action_feedback=last_feedback,
            step_reward=step_reward,
            cumulative_reward=self._state.cumulative_reward,
            done=done,
            reward=step_reward,
            metadata={
                "task_id": self._state.task_id,
                "difficulty": self._state.difficulty,
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "sla_breaches": self._state.sla_breaches,
                "consecutive_wrong": self._state.consecutive_wrong,
                "task_description": TASKS[self._state.task_id]["description"],
                "available_tasks": list_tasks(),
            },
        )
