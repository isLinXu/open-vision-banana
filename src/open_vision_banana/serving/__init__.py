"""Optional inference-serving adapters.

Serving dependencies are intentionally imported only when an application is
created, so the core package remains usable without FastAPI.
"""

from .fastapi_app import create_app

__all__ = ["create_app"]
