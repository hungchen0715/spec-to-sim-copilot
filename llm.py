"""
LLM integration for battery module assembly spec generation and repair.

Supports Gemini (primary) and OpenAI (fallback), with a local inference
mode that returns pre-computed samples for offline scenarios.

The system prompt is specifically designed for battery manufacturing:
- Cell types from catalogue (LG, Hyundai, CATL)
- Physical placement constraints
- Robot work envelope awareness
"""
import json
from schema import (
    ModuleTask, CellSpec, CellType, RobotConfig,
    RobotModel, GripperType, InspectionCamera,
)
from config import (
    GEMINI_API_KEY, OPENAI_API_KEY,
    GEMINI_MODEL, OPENAI_MODEL,
    LLM_PROVIDER, MAX_REPAIR_ATTEMPTS,
    CELL_CATALOG, ROBOT_PROFILES,
    LLAMACPP_BASE_URL, LLAMACPP_MODEL,
)


# ── System prompt: battery manufacturing specialist ──
SYSTEM_PROMPT = """You are an industrial automation engineer AI that generates battery module assembly specifications for simulation in NVIDIA Isaac Sim.

Given a natural-language task description, generate a structured ModuleTask JSON for a battery pack assembly station.

CELL CATALOGUE (only use these):
{cell_info}

ROBOT PROFILES:
{robot_info}

STRICT RULES:
1. Only use cell_type values from: {cell_types}
2. Only use robot model values from: {robot_models}
3. Only use gripper values from: Vacuum_Gripper_V1, Mechanical_Clamp_V2, Soft_Finger_V1
4. Cell positions must be within module_tray_bounds (default: 0.8m x 0.6m x 0.3m). All coordinates positive.
5. Cell IDs must be unique, format: "Cell_01", "Cell_02", etc.
6. Cell rotation_y must be 0.0 or 180.0 degrees only (busbar welding requirement).
7. Minimum center-to-center spacing between cells must respect thermal safety gaps.
8. All cell positions must be within the robot's max reach from its base_position.
9. Camera position z must be positive and above the module tray.
10. camera.look_at should point to the center of the module tray.

For a 2x3 grid of LG_E63 cells, use 0.056m center-to-center spacing (50mm cell + 6mm gap for safety margin).
For a 2x4 grid of HY_50Ah cells, use 0.052m center-to-center spacing (45mm cell + 7mm gap for safety margin).

Respond ONLY with valid JSON matching the ModuleTask schema. No markdown, no explanation."""

REPAIR_PROMPT = """The ModuleTask you generated has validation errors from the industrial safety checker. Fix them.

ORIGINAL USER REQUEST:
{original_prompt}

YOUR PREVIOUS OUTPUT:
{spec_json}

VALIDATION ERRORS (you MUST fix ALL of these):
{errors}

Fix ALL errors while preserving the user's intent. Pay careful attention to:
- Thermal safety gaps between cells
- Robot reach limits
- Cell orientation (only 0° or 180°)
- Module tray boundaries

Respond ONLY with the corrected ModuleTask JSON."""


def _get_system_prompt() -> str:
    """Build system prompt with real spec data."""
    cell_info = "\n".join(
        f"  - {name}: {spec['width']*1000:.0f}mm × {spec['depth']*1000:.0f}mm × "
        f"{spec['height']*1000:.0f}mm, {spec['weight']}kg, min gap: {spec['min_gap']*1000:.0f}mm"
        for name, spec in CELL_CATALOG.items()
    )
    robot_info = "\n".join(
        f"  - {name}: reach={spec['max_reach']}m, dead_zone={spec['dead_zone']}m, "
        f"payload={spec['payload']}kg"
        for name, spec in ROBOT_PROFILES.items()
    )
    cell_types = ", ".join(CELL_CATALOG.keys())
    robot_models = ", ".join(ROBOT_PROFILES.keys())

    return SYSTEM_PROMPT.format(
        cell_info=cell_info,
        robot_info=robot_info,
        cell_types=cell_types,
        robot_models=robot_models,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider: Gemini
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _generate_with_gemini(prompt: str) -> ModuleTask:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_get_system_prompt(),
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ModuleTask,
            temperature=0.7,
        ),
    )
    response = model.generate_content(prompt)
    data = json.loads(response.text)
    return ModuleTask(**data)


def _repair_with_gemini(original_prompt: str, spec: ModuleTask, errors: str) -> ModuleTask:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_get_system_prompt(),
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ModuleTask,
            temperature=0.3,
        ),
    )
    repair_msg = REPAIR_PROMPT.format(
        original_prompt=original_prompt,
        spec_json=spec.model_dump_json(indent=2),
        errors=errors,
    )
    response = model.generate_content(repair_msg)
    data = json.loads(response.text)
    return ModuleTask(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider: OpenAI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _generate_with_openai(prompt: str) -> ModuleTask:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    data = json.loads(response.choices[0].message.content)
    return ModuleTask(**data)


def _repair_with_openai(original_prompt: str, spec: ModuleTask, errors: str) -> ModuleTask:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    repair_msg = REPAIR_PROMPT.format(
        original_prompt=original_prompt,
        spec_json=spec.model_dump_json(indent=2),
        errors=errors,
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": repair_msg},
        ],
        temperature=0.3,
    )
    data = json.loads(response.choices[0].message.content)
    return ModuleTask(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider: Llama.cpp (local, OpenAI-compatible API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _generate_with_llamacpp(prompt: str) -> ModuleTask:
    from openai import OpenAI
    client = OpenAI(base_url=LLAMACPP_BASE_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model=LLAMACPP_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    data = json.loads(response.choices[0].message.content)
    return ModuleTask(**data)


def _repair_with_llamacpp(original_prompt: str, spec: ModuleTask, errors: str) -> ModuleTask:
    from openai import OpenAI
    client = OpenAI(base_url=LLAMACPP_BASE_URL, api_key="not-needed")
    repair_msg = REPAIR_PROMPT.format(
        original_prompt=original_prompt,
        spec_json=spec.model_dump_json(indent=2),
        errors=errors,
    )
    response = client.chat.completions.create(
        model=LLAMACPP_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": repair_msg},
        ],
        temperature=0.3,
    )
    data = json.loads(response.choices[0].message.content)
    return ModuleTask(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider: Demo (offline fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Scenario A: Valid 2x3 LG module (everything correct)
_DEMO_VALID = ModuleTask(
    task_id="BatteryModule_2x3_LG",
    description="2x3 LG E63 cell module assembly with UR10e robot arm",
    cells=[
        CellSpec(id="Cell_01", cell_type=CellType.LG_E63,
                 position=[0.200, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_02", cell_type=CellType.LG_E63,
                 position=[0.256, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_03", cell_type=CellType.LG_E63,
                 position=[0.312, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_04", cell_type=CellType.LG_E63,
                 position=[0.200, 0.280, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_05", cell_type=CellType.LG_E63,
                 position=[0.256, 0.280, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_06", cell_type=CellType.LG_E63,
                 position=[0.312, 0.280, 0.0], rotation_y=0.0),
    ],
    robot=RobotConfig(
        model=RobotModel.UR10e,
        base_position=[0.0, 0.0, 0.0],
        gripper=GripperType.VACUUM,
    ),
    camera=InspectionCamera(
        position=[0.4, 0.3, 0.6],
        look_at=[0.4, 0.3, 0.0],
        fov=60.0,
    ),
    module_tray_bounds=[0.8, 0.6, 0.3],
)

# Scenario B: Overcrowded — 20 HY cells packed too tightly, some out of tray
_DEMO_CROWDED = ModuleTask(
    task_id="BatteryModule_20_HY_Dense",
    description="20 HY 50Ah cells packed in single row — intentionally overcrowded",
    cells=[
        CellSpec(id=f"Cell_{i:02d}", cell_type=CellType.HY_50Ah,
                 position=[0.050 + i * 0.040, 0.150, 0.0], rotation_y=0.0)
        for i in range(20)
    ],
    robot=RobotConfig(
        model=RobotModel.KUKA_KR6,
        base_position=[0.0, 0.0, 0.0],
        gripper=GripperType.CLAMP,
    ),
    camera=InspectionCamera(
        position=[0.4, 0.3, 0.6],
        look_at=[0.4, 0.3, 0.0],
        fov=75.0,
    ),
    module_tray_bounds=[0.5, 0.3, 0.3],
)

# Scenario C: Bad placement — Cell_05 out of reach, Cell_03 wrong rotation
_DEMO_BAD = ModuleTask(
    task_id="CATL_Module_BadPlacement",
    description="6 CATL LFP cells with intentional placement and rotation errors",
    cells=[
        CellSpec(id="Cell_01", cell_type=CellType.CATL_LFP,
                 position=[0.200, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_02", cell_type=CellType.CATL_LFP,
                 position=[0.260, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_03", cell_type=CellType.CATL_LFP,
                 position=[0.320, 0.150, 0.0], rotation_y=45.0),     # ← BAD: 45°
        CellSpec(id="Cell_04", cell_type=CellType.CATL_LFP,
                 position=[0.200, 0.330, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_05", cell_type=CellType.CATL_LFP,
                 position=[2.500, 1.000, 0.0], rotation_y=0.0),      # ← OUT OF REACH
        CellSpec(id="Cell_06", cell_type=CellType.CATL_LFP,
                 position=[0.320, 0.330, 0.0], rotation_y=0.0),
    ],
    robot=RobotConfig(
        model=RobotModel.FANUC_CRX10,
        base_position=[0.0, 0.0, 0.0],
        gripper=GripperType.VACUUM,
    ),
    camera=InspectionCamera(
        position=[0.4, 0.3, 0.6],
        look_at=[0.4, 0.3, 0.0],
        fov=60.0,
    ),
)

# Scenario C repaired: Cell_03 rotation fixed, Cell_05 moved within reach
_DEMO_BAD_REPAIRED = ModuleTask(
    task_id="CATL_Module_Repaired",
    description="6 CATL LFP cells — repaired: rotation fixed, placement corrected",
    cells=[
        CellSpec(id="Cell_01", cell_type=CellType.CATL_LFP,
                 position=[0.200, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_02", cell_type=CellType.CATL_LFP,
                 position=[0.260, 0.150, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_03", cell_type=CellType.CATL_LFP,
                 position=[0.320, 0.150, 0.0], rotation_y=0.0),      # ← FIXED: 0°
        CellSpec(id="Cell_04", cell_type=CellType.CATL_LFP,
                 position=[0.200, 0.330, 0.0], rotation_y=0.0),
        CellSpec(id="Cell_05", cell_type=CellType.CATL_LFP,
                 position=[0.380, 0.330, 0.0], rotation_y=0.0),      # ← FIXED: moved close
        CellSpec(id="Cell_06", cell_type=CellType.CATL_LFP,
                 position=[0.320, 0.330, 0.0], rotation_y=0.0),
    ],
    robot=RobotConfig(
        model=RobotModel.FANUC_CRX10,
        base_position=[0.0, 0.0, 0.0],
        gripper=GripperType.VACUUM,
    ),
    camera=InspectionCamera(
        position=[0.4, 0.3, 0.6],
        look_at=[0.4, 0.3, 0.0],
        fov=60.0,
    ),
)


def _detect_demo_scenario(prompt: str) -> str:
    """Match the prompt to one of the 3 demo scenarios."""
    p = prompt.lower()
    # Scenario B keywords: overcrowded / packed / 20 / tightly / single row
    if any(kw in p for kw in ["20", "tightly", "packed", "single row", "crowded", "dense"]):
        return "crowded"
    # Scenario C keywords: far / rotate / 45 / out of reach / Cell_05
    if any(kw in p for kw in ["45", "far", "rotate", "cell_05", "out of reach", "2.5"]):
        return "bad"
    # Default: valid scenario
    return "valid"


def _generate_with_demo(prompt: str) -> ModuleTask:
    scenario = _detect_demo_scenario(prompt)
    if scenario == "crowded":
        return _DEMO_CROWDED
    elif scenario == "bad":
        return _DEMO_BAD
    return _DEMO_VALID


def _repair_with_demo(original_prompt: str, spec: ModuleTask, errors: str) -> ModuleTask:
    scenario = _detect_demo_scenario(original_prompt)
    if scenario == "bad":
        return _DEMO_BAD_REPAIRED
    # For crowded scenario, return valid as a "simplified" repair
    return _DEMO_VALID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _is_llamacpp_available() -> bool:
    """Check if llama.cpp server is reachable."""
    try:
        import urllib.request
        url = LLAMACPP_BASE_URL.replace("/v1", "/health")
        req = urllib.request.urlopen(url, timeout=2)
        return req.status == 200
    except Exception:
        return False


def _get_provider():
    provider = LLM_PROVIDER.lower()
    if provider == "llamacpp" and _is_llamacpp_available():
        return "llamacpp"
    elif provider == "gemini" and GEMINI_API_KEY:
        return "gemini"
    elif provider == "openai" and OPENAI_API_KEY:
        return "openai"
    elif _is_llamacpp_available():
        return "llamacpp"
    elif GEMINI_API_KEY:
        return "gemini"
    elif OPENAI_API_KEY:
        return "openai"
    else:
        return "local"


def generate_task_spec(prompt: str) -> tuple[ModuleTask, str]:
    """Generate a ModuleTask from a natural-language prompt."""
    provider = _get_provider()
    generators = {
        "llamacpp": _generate_with_llamacpp,
        "gemini": _generate_with_gemini,
        "openai": _generate_with_openai,
        "local": _generate_with_demo,
    }
    try:
        spec = generators[provider](prompt)
        return spec, provider
    except Exception as e:
        print(f"[llm] {provider} failed: {e}")
        for fallback in ["llamacpp", "gemini", "openai", "local"]:
            if fallback == provider:
                continue
            try:
                fb_gen = generators.get(fallback)
                if fallback == "local" or (
                    fallback == "llamacpp" and _is_llamacpp_available()
                ) or (
                    fallback == "gemini" and GEMINI_API_KEY
                ) or (
                    fallback == "openai" and OPENAI_API_KEY
                ):
                    spec = fb_gen(prompt)
                    return spec, f"{fallback} (fallback)"
            except Exception:
                continue
        return _generate_with_demo(prompt), "local (fallback)"


def repair_task_spec(
    original_prompt: str,
    spec: ModuleTask,
    errors_text: str,
) -> tuple[ModuleTask, str]:
    """Ask the LLM to fix validation errors."""
    provider = _get_provider()
    repairers = {
        "llamacpp": _repair_with_llamacpp,
        "gemini": _repair_with_gemini,
        "openai": _repair_with_openai,
        "local": _repair_with_demo,
    }
    try:
        repairer = repairers[provider]
        new_spec = repairer(original_prompt, spec, errors_text)
        return new_spec, provider
    except Exception as e:
        print(f"[llm] repair with {provider} failed: {e}")
        return spec, f"{provider} (repair failed)"
