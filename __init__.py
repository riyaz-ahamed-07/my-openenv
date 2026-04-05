"""
SupportTriageEnv — OpenEnv-compliant Customer Support Triage Environment.

Public API:
    SupportTriageAction       — action model (priority + department + response)
    SupportTriageObservation  — observation model (ticket + SLA + feedback)
    SupportTriageState        — state model (episode metadata)
    SupportTriageEnv          — EnvClient subclass

Example:
    import asyncio
    from support_triage_env import SupportTriageAction, SupportTriageEnv

    async def main():
        async with SupportTriageEnv(base_url="https://your-space.hf.space") as env:
            result = await env.reset(task_id="task_3_hard")
            obs = result.observation
            print(obs.ticket_text)

    asyncio.run(main())
"""

from models import SupportTriageAction, SupportTriageObservation, SupportTriageState
from client import SupportTriageEnv

__all__ = [
    "SupportTriageAction",
    "SupportTriageObservation",
    "SupportTriageState",
    "SupportTriageEnv",
]

__version__ = "1.0.0"
