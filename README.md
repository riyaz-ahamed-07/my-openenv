---
title: Support Triage Env Environment Server
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - llm-agents
---

# 🎫 SupportTriageEnv

**An OpenEnv environment for training AI agents on real-world customer support ticket triage.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://www.docker.com/)

---

## Overview

SupportTriageEnv simulates an **enterprise customer support inbox** — one of the highest-volume AI deployment domains today. An AI agent operates as a support team lead, triaging incoming tickets by:

1. **Classifying priority** — urgent / high / medium / low
2. **Routing to the correct department** — billing / technical / account / general
3. **Drafting professional responses** — scored against a structured quality rubric

This environment models real challenges professionals face: time pressure (SLA deadlines), varying customer importance (tier weighting), and escalating penalties for repeated mistakes.

---

## Why This Environment?

| Criterion | SupportTriageEnv |
|---|---|
| Real-world utility | ✅ Top LLM deployment domain (millions of tickets/day) |
| Novel | ✅ No existing OpenEnv triage environment |
| Stateful | ✅ SLA countdowns, customer context, compounding penalties |
| Dense rewards | ✅ Every step yields a meaningful signal |
| Deterministic | ✅ No external APIs inside the environment |
| Difficulty range | ✅ Easy → Hard with genuine progression |

---

## Action & Observation Spaces

### Action Space (`SupportTriageAction`)

| Field | Type | Values | Description |
|---|---|---|---|
| `ticket_id` | `str` | — | ID of ticket being triaged |
| `priority` | `str` | `urgent \| high \| medium \| low` | Urgency classification |
| `department` | `str` | `billing \| technical \| account \| general` | Routing target |
| `response_draft` | `str` | 0–2000 chars | Draft response (scored for Task 3) |

### Observation Space (`SupportTriageObservation`)

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Current ticket's unique ID |
| `ticket_text` | `str` | Full text of the support ticket |
| `customer_tier` | `str` | `enterprise \| pro \| free` (affects reward weight) |
| `previous_interactions` | `int` | Number of prior contacts from this customer |
| `is_repeat_complaint` | `bool` | True if same issue reported before |
| `sla_deadline_steps` | `int` | Steps before SLA expires (0 = already breached) |
| `remaining_tickets` | `int` | Tickets left in episode queue |
| `last_action_feedback` | `dict \| None` | Grading breakdown from previous step |
| `step_reward` | `float \| None` | Reward earned in last step |
| `cumulative_reward` | `float` | Episode total so far |
| `done` | `bool` | Whether episode has ended |

---

## Tasks

### Task 1 — Easy: Single-Ticket Priority Classification

**Objective:** Correctly classify the priority of a single support ticket.

- **Tickets:** 1 (deterministic)
- **Max steps:** 1
- **Scored fields:** `priority` only
- **Reward design:**
  - Exact match: `+1.0`
  - Adjacent level (e.g., high for urgent): `+0.5`
  - Far miss: `0.0`
  - Missed urgent: `-0.30` penalty

**Expected baseline score:** ~0.70

---

### Task 2 — Medium: Multi-Ticket Routing & Priority

**Objective:** Process 5 tickets, classifying priority AND routing each correctly.

- **Tickets:** 5 (includes enterprise customers, repeat complaints, and SLA pressure)
- **Max steps:** 5
- **Scored fields:** `priority` (50%) + `department` (50%)
- **Reward design:**
  - Per-step: `0.5 × priority_score + 0.5 × routing_score`
  - Enterprise tier adds a bonus (+tier_bonus on base score)
  - SLA breach: `-0.15` per expired ticket
  - Consecutive wrong answers: `-0.05 × consecutive_count` compounding

**Expected baseline score:** ~0.55

---

### Task 3 — Hard: Full Triage Pipeline with Response Drafting

**Objective:** Process 3 complex enterprise tickets with complete triage including a written response.

- **Tickets:** 3 (all enterprise, all with tight SLAs, multi-issue scenarios)
- **Max steps:** 3
- **Scored fields:** `priority` (25%) + `department` (25%) + `response_draft` (50%)
- **Response quality rubric (deterministic, no LLM):**

| Component | Max Score | Criteria |
|---|---|---|
| Acknowledges issue | +0.20 | References ticket keywords + acknowledgement signals |
| Concrete resolution steps | +0.30 | ≥2 action words, structured steps preferred |
| Department context | +0.20 | Uses domain-appropriate terminology |
| Professional tone | +0.20 | Warm, direct; no generic clichés |
| Conciseness | +0.10 | 30–150 words (ideal range) |
| **Penalties** | | |
| Vague/empty response | -0.20 | <20 words or no actionable content |
| Overly long | -0.10 | >500 words |
| Discouraged phrases | -0.10 | Generic "valued customer", "as soon as possible" |

**Expected baseline score:** ~0.35

---

## Reward Function

### Reward Shaping

```
Per-step reward = (
    w_priority × priority_score +
    w_routing  × routing_score  +
    w_response × response_score
) - priority_penalty - routing_penalty - sla_penalty - compounding_penalty + tier_bonus
```

Clamped to `[0.0, 1.0]` per step.

### Penalty Schedule

| Event | Penalty |
|---|---|
| Missed urgent ticket | -0.30 |
| Wrong department | -0.20 |
| SLA breach (expired deadline) | -0.15 |
| High-risk combo miss (urgent+technical) | -0.15 extra |
| Compounding consecutive errors | -0.05 × count |
| Empty/vague response (Task 3) | -0.20 |
| Overly long response (Task 3) | -0.10 |

### Customer Tier Multipliers

| Tier | Multiplier |
|---|---|
| `enterprise` | 2.0× (bonus applied) |
| `pro` | 1.5× (bonus applied) |
| `free` | 1.0× (no bonus) |

### SLA System

Each ticket has an `sla_deadline_steps` counter that counts down as the episode progresses. Handling tickets out of order causes SLA breaches on the delayed tickets, incentivising the agent to prioritise correctly.

---

## Episode Boundaries

- `reset(task_id=...)` → starts a fresh episode, returns first ticket
- `step(action)` → grades action, returns next ticket or `done=True`
- Episode ends when all tickets are processed (`done=True`)
- `state()` → returns episode metadata (step count, SLA breaches, cumulative reward, etc.)

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- `pip install openenv-core`

### Local Development (No Docker)

```bash
git clone https://github.com/your-username/support-triage-env
cd support-triage-env

# Install dependencies
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` to see the FastAPI docs.  
Open `http://localhost:8000/web` to use the interactive web interface.

### Docker

```bash
# Build
docker build -t support-triage-env .

# Run
docker run -p 8000:8000 support-triage-env

# Health check
curl http://localhost:8000/health
```

### Python Client

```python
import asyncio
from client import SupportTriageEnv
from models import SupportTriageAction

async def main():
    async with SupportTriageEnv(base_url="http://localhost:8000") as env:
        # Task 2: multi-ticket routing
        result = await env.reset(task_id="task_2_medium")

        while not result.done:
            obs = result.observation
            print(f"\nTicket {obs.ticket_id}: {obs.ticket_text[:80]}...")
            print(f"SLA: {obs.sla_deadline_steps} steps | Tier: {obs.customer_tier}")

            action = SupportTriageAction(
                ticket_id=obs.ticket_id,
                priority="high",
                department="technical",
                response_draft="",
            )
            result = await env.step(action)
            print(f"Reward: {result.reward:.3f}")

asyncio.run(main())
```

### Sync Usage

```python
from client import SupportTriageEnv
from models import SupportTriageAction

with SupportTriageEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="task_1_easy")
    obs = result.observation
    action = SupportTriageAction(
        ticket_id=obs.ticket_id,
        priority="urgent",
        department="technical",
        response_draft="",
    )
    result = env.step(action)
    print(f"Score: {result.reward}")
```

---

## Running the Baseline Inference Script

```bash
# Set required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-api-key"
export HF_SPACE_URL="http://localhost:8000"  # or your HF Space URL

# Run baseline
python inference.py
```

### Expected Output Format

```json
{"event": "START", "task": "task_1_easy", "env": "support_triage_env", "model": "gpt-4o-mini", ...}
{"event": "STEP", "step": 1, "action": "priority=urgent, dept=technical", "reward": 0.85, "done": true, ...}
{"event": "END", "success": true, "steps": 1, "score": 0.85, "rewards": [0.85], ...}
```

### Expected Baseline Scores

| Task | Model | Expected Score |
|---|---|---|
| task_1_easy | gpt-4o-mini | ~0.70 |
| task_2_medium | gpt-4o-mini | ~0.55 |
| task_3_hard | gpt-4o-mini | ~0.35 |
| **Average** | | **~0.53** |

---

## HuggingFace Spaces Deployment

```bash
pip install openenv-core

# Login to Hugging Face
huggingface-cli login

# Push to HF Spaces
openenv push --repo-id your-username/support-triage-env
```

Once deployed, set `HF_SPACE_URL=https://your-username-support-triage-env.hf.space` in your inference script.

---

## Project Structure

```
support-triage-env/
├── openenv.yaml              # OpenEnv manifest (spec_version: 1)
├── Dockerfile                # Multi-stage Docker build
├── pyproject.toml            # Package metadata & dependencies
├── inference.py              # Baseline inference script (root level, required)
├── README.md                 # This file
├── models.py                 # Pydantic Action, Observation, State
├── client.py                 # SupportTriageEnv(EnvClient)
├── __init__.py               # Public API exports
├── tasks/
│   ├── ticket_corpus.py      # Synthetic deterministic ticket dataset (17 tickets)
│   ├── task_registry.py      # Task definitions + Episode class
│   ├── grader.py             # Deterministic rubric grader
│   └── __init__.py
└── server/
    ├── support_triage_environment.py  # SupportTriageEnvironment(Environment)
    ├── app.py                         # FastAPI app via create_app()
    ├── requirements.txt               # Server dependencies
    └── __init__.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `POST` | `/reset` | Start episode → initial observation |
| `POST` | `/step` | Execute action → observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/schema` | Action/Observation JSON schemas |
| `WS` | `/ws` | WebSocket for real-time interaction |
| `GET` | `/web` | Interactive web UI (if ENABLE_WEB_INTERFACE=true) |

### Reset Payload

```json
{
  "task_id": "task_2_medium",
  "seed": null,
  "episode_id": "my-episode-001"
}
```

### Step Payload

```json
{
  "action": {
    "ticket_id": "T101",
    "priority": "urgent",
    "department": "technical",
    "response_draft": "I understand your data export has been running for 6 hours without results. This is being escalated to our infrastructure team immediately as a priority issue. We will investigate the pipeline logs and contact you within 30 minutes with an update.",
    "metadata": {}
  }
}
```

---

## License

MIT License — see [LICENSE](LICENSE) file.