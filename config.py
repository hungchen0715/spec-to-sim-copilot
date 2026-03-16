"""
Configuration for the AI-to-USD Battery Factory Pipeline.
Loads API keys from environment and defines industrial constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider Settings ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# ── Llama.cpp Server Settings ──
# llama.cpp exposes an OpenAI-compatible API
LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "local-model")

# ── Repair Loop ──
MAX_REPAIR_ATTEMPTS = 3

# ── Industrial Robot Configurations ──
# Real-world specs from manufacturer datasheets
ROBOT_PROFILES = {
    "UR10e": {
        "max_reach": 1.30,      # meters
        "dead_zone": 0.18,      # meters (base collision radius)
        "payload": 12.5,        # kg
        "base_position": [0.0, 0.0, 0.0],
    },
    "KUKA_KR6": {
        "max_reach": 0.90,
        "dead_zone": 0.15,
        "payload": 6.0,
        "base_position": [0.0, 0.0, 0.0],
    },
    "FANUC_CRX10": {
        "max_reach": 1.25,
        "dead_zone": 0.20,
        "payload": 10.0,
        "base_position": [0.0, 0.0, 0.0],
    },
}
DEFAULT_ROBOT = "UR10e"

# ── Battery Cell Specifications ──
# Based on real EV battery cell dimensions
CELL_CATALOG = {
    "LG_E63": {
        "width": 0.050,     # 50mm pouch cell width
        "depth": 0.120,     # 120mm
        "height": 0.200,    # 200mm
        "weight": 0.82,     # kg
        "min_gap": 0.002,   # 2mm thermal safety gap
    },
    "HY_50Ah": {
        "width": 0.045,
        "depth": 0.148,
        "height": 0.095,
        "weight": 1.06,
        "min_gap": 0.003,   # 3mm gap for prismatic cell cooling
    },
    "CATL_LFP": {
        "width": 0.054,
        "depth": 0.174,
        "height": 0.207,
        "weight": 1.24,
        "min_gap": 0.002,
    },
}

# ── Workspace / Module Tray Limits ──
MODULE_TRAY_BOUNDS = [0.8, 0.6, 0.3]  # meters [x, y, z]
MAX_CELLS_PER_MODULE = 24
MAX_CELLS_WARNING = 16

# ── Preview Settings ──
PREVIEW_DPI = 120
PREVIEW_FIGSIZE = (10, 8)

# ── Gripper Types ──
GRIPPER_TYPES = [
    "Vacuum_Gripper_V1",
    "Mechanical_Clamp_V2",
    "Soft_Finger_V1",
]
