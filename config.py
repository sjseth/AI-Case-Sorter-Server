"""User-editable configuration for the CaseSorter AI Server.

Edit the values below to control where the server listens, which trained
ConvNeXt checkpoints it serves, and (optionally) the API key it requires.
The CaseSorter desktop client points at this server by setting its OpenAI
endpoint to "http://<host>:<port>" and its OpenAI model name to one of the
aliases defined in MODELS.
"""

from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

# Interface the server binds to.
#   "127.0.0.1" -> localhost only (safest; reachable only from this machine)
#   "0.0.0.0"   -> all interfaces (reachable from other machines on the LAN)
HOST: str = "127.0.0.1"

# TCP port to listen on.
PORT: int = 8000


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

# Optional Bearer token. Leave as None (or empty string) to accept any caller
# without authentication. When set, clients must send
#   Authorization: Bearer <API_KEY>
# on every request. Strongly recommended when HOST is "0.0.0.0".
API_KEY: Optional[str] = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Aliases the OpenAI request's "model" field can use, mapped to the on-disk
# .pth checkpoint to load. Paths may be absolute or relative to the directory
# this file lives in.
#
# Example:
#   MODELS = {
#       "headstamps-v3":        r"C:\caselib\models\headstamps_v3.pth",
#       "primers-experimental": "models/primers_swa.pth",
#   }
MODELS: Dict[str, str] = {
    # "headstamps-v3": "models/headstamps_v3.pth",
}

# Optional per-model tweaks. Keys must match aliases in MODELS.
# Currently supported sub-keys:
#   "image_size" -> int. Resize edge used at inference. Defaults to 224
#                   (the torchvision IMAGENET1K_V1 default for ConvNeXt).
MODEL_OPTIONS: Dict[str, dict] = {
    # "headstamps-v3": {"image_size": 232},
}


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

# When True, every model in MODELS is loaded at startup so the first request
# does not pay the load cost. When False, models load lazily on first use.
PRELOAD_MODELS: bool = False

# Logging level. One of DEBUG / INFO / WARNING / ERROR.
LOG_LEVEL: str = "INFO"
