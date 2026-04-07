"""
inference.py — Baseline Inference Script for SupportTriageEnv

Runs a language model agent against all 3 tasks and emits structured
stdout logs in the required [START] / [STEP] / [END] format.

Environment variables (required):
    API_BASE_URL    OpenAI-compatible API base URL
                    e.g. "https://api.openai.com/v1" or local Ollama endpoint
    MODEL_NAME      Model identifier, e.g. "gpt-4o-mini", "llama3.2"
    HF_TOKEN        API key (used as OPENAI_API_KEY)
    HF_SPACE_URL    (optional) HuggingFace Space URL for the deployed env
                    If not set, falls back to http://localhost:8000

Usage:
    API_BASE_URL=https://api.openai.com/v1 \\
    MODEL_NAME=gpt-4o-mini \\
    HF_TOKEN=sk-... \\
    python inference.py

Expected baseline scores (approximate):
    task_1_easy   : ~0.70
    task_2_medium : ~0.55
    task_3_hard   : ~0.35
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Checklist requires no default for HF_TOKEN
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
HF_SPACE_URL: str = os.environ.get("HF_SPACE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME: Optional[str] = os.environ.get("LOCAL_IMAGE_NAME")

# For local OpenAI client compatibility if HF_TOKEN is missing
API_KEY: str = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")

TEMPERATURE: float = 0.2          # Low temp for reproducibility
MAX_TOKENS: int = 512             # Cap response length
MAX_STEPS_PER_TASK: int = 10      # Safety cap (each task has its own max)
SUCCESS_SCORE_THRESHOLD: float = 0.5

# Tasks to run (in order)
TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]

BENCHMARK = "support_triage_env"
IMAGE_NAME = "support-triage-env"   # Docker image name if running locally

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support triage agent. Your job is to:
1. Read the support ticket carefully
2. Classify its priority: urgent | high | medium | low
3. Route it to the correct department: billing | technical | account | general
4. Draft a professional response (required for hard task)

Priority guidelines:
- urgent: production outages, security breaches, complete service failures, SLA at risk
- high: significant functionality broken, data issues, financial discrepancies
- medium: degraded service, features not working, account issues
- low: general questions, cosmetic issues, minor feature requests

Department guidelines:
- billing: payment failures, invoices, refunds, subscription changes, pricing
- technical: bugs, API issues, integration failures, outages, performance
- account: login, password, permissions, team management, settings, email
- general: product questions, documentation, feature requests, feedback

You MUST respond with valid JSON in this exact format:
{
  "ticket_id": "<id from the observation>",
  "priority": "<urgent|high|medium|low>",
  "department": "<billing|technical|account|general>",
  "response_draft": "<your response to the customer, 50-150 words>"
}

For task_1_easy: only priority matters, but provide all fields.
For task_2_medium: priority + department matter, response_draft can be brief.
For task_3_hard: ALL fields are scored. Write a genuine, helpful response.

Be direct, professional, and concise. Do not add JSON markdown fences."""

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers — STRICT format required by evaluator
# ─────────────────────────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Any,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────


def build_user_prompt(task_id: str, obs_dict: Dict[str, Any], step: int, history: List[str]) -> str:
    """Construct the user message for the model from the current observation."""
    ticket_id = obs_dict.get("ticket_id", "unknown")
    ticket_text = obs_dict.get("ticket_text", "")
    customer_tier = obs_dict.get("customer_tier", "free")
    prev_interactions = obs_dict.get("previous_interactions", 0)
    is_repeat = obs_dict.get("is_repeat_complaint", False)
    sla_steps = obs_dict.get("sla_deadline_steps", 99)
    remaining = obs_dict.get("remaining_tickets", 1)
    feedback = obs_dict.get("last_action_feedback")

    parts = [
        f"=== SUPPORT TICKET (Task: {task_id}, Step {step}) ===",
        f"Ticket ID: {ticket_id}",
        f"Customer Tier: {customer_tier}",
        f"Previous Interactions: {prev_interactions}",
        f"Repeat Complaint: {is_repeat}",
        f"SLA Deadline: {sla_steps} steps remaining (URGENT if ≤ 1)",
        f"Remaining Tickets in Queue: {remaining}",
        "",
        "--- TICKET CONTENT ---",
        ticket_text,
    ]

    if feedback:
        parts += [
            "",
            "--- PREVIOUS STEP FEEDBACK ---",
            f"Priority Score: {feedback.get('priority_score', 'N/A')}",
            f"Routing Score: {feedback.get('routing_score', 'N/A')}",
            f"Response Score: {feedback.get('response_score', 'N/A')}",
            f"Step Reward: {feedback.get('total_reward', 'N/A')}",
            f"Notes: {'; '.join(feedback.get('notes', []))}",
        ]

    if history:
        parts += ["", "--- HISTORY (last 3 steps) ---"]
        parts += history[-3:]

    parts += [
        "",
        "--- YOUR TASK ---",
        f"Triage this ticket. Respond with JSON only.",
    ]

    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    task_id: str,
    obs_dict: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call the model and parse the JSON action response."""
    user_prompt = build_user_prompt(task_id, obs_dict, step, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model adds them
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        action = json.loads(text)
        return action

    except json.JSONDecodeError:
        # Fallback: return a safe default action
        ticket_id = obs_dict.get("ticket_id", "unknown")
        print(f"[DEBUG] JSON parse failed, using fallback action", flush=True)
        return {
            "ticket_id": ticket_id,
            "priority": "medium",
            "department": "general",
            "response_draft": (
                "Thank you for reaching out. I understand your concern and "
                "will escalate this to the appropriate team immediately. "
                "You will hear back shortly."
            ),
        }
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        ticket_id = obs_dict.get("ticket_id", "unknown")
        return {
            "ticket_id": ticket_id,
            "priority": "medium",
            "department": "general",
            "response_draft": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_task(
    client: OpenAI,
    task_id: str,
) -> Dict[str, Any]:
    """
    Run a single task against the environment.

    Returns a summary dict with score, rewards, and steps taken.
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    from client import SupportTriageEnv
    from models import SupportTriageAction

    history: List[str] = []
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    try:
        async with SupportTriageEnv(base_url=HF_SPACE_URL) as env:
            # Reset to start the episode
            result = await env.reset(task_id=task_id)

            max_possible_steps = MAX_STEPS_PER_TASK
            step = 0

            while not result.done and step < max_possible_steps:
                step += 1

                # Extract observation
                obs = result.observation
                if hasattr(obs, "model_dump"):
                    obs_dict = obs.model_dump()
                elif hasattr(obs, "dict"):
                    obs_dict = obs.dict()
                else:
                    obs_dict = dict(obs)

                # Get model action
                action_dict = get_model_action(client, task_id, obs_dict, step, history)

                # Build typed action
                try:
                    action = SupportTriageAction(
                        ticket_id=action_dict.get("ticket_id", obs_dict.get("ticket_id", "unknown")),
                        priority=action_dict.get("priority", "medium"),
                        department=action_dict.get("department", "general"),
                        response_draft=action_dict.get("response_draft", ""),
                    )
                except Exception as e:
                    print(f"[DEBUG] Action validation error: {e}", flush=True)
                    action = SupportTriageAction(
                        ticket_id=obs_dict.get("ticket_id", "unknown"),
                        priority="medium",
                        department="general",
                        response_draft="",
                    )

                # Execute step
                result = await env.step(action)

                reward = float(result.reward or 0.0)
                done = result.done
                rewards.append(reward)
                steps_taken = step

                # Log step
                action_summary = (
                    f"priority={action.priority}, dept={action.department}, "
                    f"response_len={len(action.response_draft)}"
                )
                log_step(step=step, action=action_summary, reward=reward, done=done, error=None)

                history.append(
                    f"Step {step}: {action_summary} → reward {reward:+.4f}"
                )

                if done:
                    break

        # Compute normalized score
        # Max possible reward = number of tickets (each ticket can earn up to 1.0)
        task_ticket_counts = {"task_1_easy": 1, "task_2_medium": 5, "task_3_hard": 3}
        max_reward = float(task_ticket_counts.get(task_id, 1))
        score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "rewards": rewards,
        "steps": steps_taken,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    print(f"[INFO] Starting SupportTriageEnv baseline", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] API Base: {API_BASE_URL}", flush=True)
    print(f"[INFO] Environment URL: {HF_SPACE_URL}", flush=True)
    print(f"[INFO] Tasks: {TASK_IDS}", flush=True)
    print("", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    for task_id in TASK_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        result = await run_task(client, task_id)
        all_results.append(result)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] Baseline Results", flush=True)
    print(f"{'='*60}", flush=True)
    overall_score = sum(r["score"] for r in all_results) / len(all_results)
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  {status} {r['task_id']:<20} score={r['score']:.4f}  "
            f"steps={r['steps']}  rewards={[round(x, 3) for x in r['rewards']]}",
            flush=True,
        )
    print(f"\n  Overall avg score: {overall_score:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
