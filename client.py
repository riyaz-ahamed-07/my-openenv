"""
SupportTriageEnv Client.

Provides the EnvClient subclass for communicating with a running
SupportTriageEnvironment server (local Docker or HF Space).

Usage:
    import asyncio
    from support_triage_env import SupportTriageAction, SupportTriageEnv

    async def main():
        async with SupportTriageEnv(base_url="http://localhost:8000") as env:
            # Start a medium-difficulty episode
            result = await env.reset(task_id="task_2_medium")
            obs = result.observation

            while not result.done:
                action = SupportTriageAction(
                    ticket_id=obs.ticket_id,
                    priority="high",
                    department="technical",
                    response_draft="",
                )
                result = await env.step(action)
                print(f"Reward: {result.reward:.3f}")

    asyncio.run(main())

Sync usage:
    with SupportTriageEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="task_1_easy")
        result = env.step(action)
"""

from typing import Any, Dict, Optional

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import StepResult
except ImportError:
    # Minimal stub when openenv-core is not available
    class EnvClient:  # type: ignore[no-redef]
        def __init__(self, base_url: str, **kwargs: Any):
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def reset(self, **kwargs):
            raise NotImplementedError("Install openenv-core to use the client")

        async def step(self, action):
            raise NotImplementedError("Install openenv-core to use the client")

        def sync(self):
            raise NotImplementedError("Install openenv-core to use the client")

from models import SupportTriageAction, SupportTriageObservation, SupportTriageState


class SupportTriageEnv(EnvClient):
    """
    Client for SupportTriageEnvironment.

    Connects to a running SupportTriageEnvironment server via WebSocket.

    Args:
        base_url:   URL of the running server (e.g. "http://localhost:8000"
                    or "https://your-space.hf.space")
        **kwargs:   Passed to EnvClient base class

    Class method:
        from_docker_image(image_name) — spin up a local Docker container
                                         and connect to it automatically.
    """

    action_type = SupportTriageAction
    observation_type = SupportTriageObservation

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(
            base_url=base_url,
            action_type=SupportTriageAction,
            observation_type=SupportTriageObservation,
            **kwargs,
        )

    async def reset(
        self,
        task_id: str = "task_1_easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Reset the environment and start a new episode.

        Args:
            task_id:    "task_1_easy" | "task_2_medium" | "task_3_hard"
            seed:       Random seed (unused — corpus is deterministic)
            episode_id: Custom episode ID

        Returns:
            StepResult with initial observation
        """
        params: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            params["seed"] = seed
        if episode_id is not None:
            params["episode_id"] = episode_id
        params.update(kwargs)
        return await super().reset(**params)
