"""
FastAPI application for SupportTriageEnv.

Creates the HTTP + WebSocket server using openenv-core's create_app factory.
This module is the entry point for both local development and Docker deployment.

Usage:
    # Local development
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

    # Production (Docker)
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # HuggingFace Space
    # CMD in Dockerfile sets this automatically
"""

import sys
import os

# Ensure imports work whether run from repo root or /app/env inside Docker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core not installed. Run: pip install openenv-core"
    ) from e

from models import SupportTriageAction, SupportTriageObservation
from server.support_triage_environment import SupportTriageEnvironment

# Create the FastAPI app
# Pass the class (not an instance) so each WebSocket session gets its own env
app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage_env",
)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
