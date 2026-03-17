"""
USD (USDA) scene exporter for validated battery module assembly specs.

Converts a validated ModuleTask into a human-readable .usda file that can be
loaded in NVIDIA Isaac Sim, Omniverse, or any USD-compatible viewer.

Design decision: Uses pure-text USDA template generation rather than the pxr
Python SDK. This avoids the heavy OpenUSD build dependency while producing
files that are 100% compatible with any USD reader.
"""
from pathlib import Path
from datetime import datetime

from schema import ModuleTask, CellSpec
from config import CELL_CATALOG, ROBOT_PROFILES


# ── USDA Templates ──

_USDA_HEADER = '''\
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
    doc = """
        Auto-generated battery module assembly scene.
        Created by: spec-to-sim-copilot
        Task: {task_id}
        Date: {date}
    """
)

def Xform "World"
{{
{children}
}}
'''

_LIGHT_TEMPLATE = '''
    def DomeLight "AmbientLight"
    {
        float inputs:intensity = 1000
        color3f inputs:color = (1.0, 0.98, 0.95)
    }

    def DistantLight "KeyLight"
    {
        float inputs:angle = 1.0
        color3f inputs:color = (1.0, 0.96, 0.90)
        float inputs:intensity = 2500
        float3 xformOp:rotateXYZ = (315, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
'''

_GROUND_TEMPLATE = '''
    def Mesh "GroundPlane"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)]
        color3f[] primvars:displayColor = [(0.75, 0.78, 0.80)]
    }
'''

_TRAY_TEMPLATE = '''
    def Cube "ModuleTray"
    {{
        float3 xformOp:translate = ({cx}, {cy}, {cz})
        float3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({r}, {g}, {b})]
    }}
'''

_CELL_TEMPLATE = '''
    def Cube "{cell_id}"
    {{
        float3 xformOp:translate = ({x}, {y}, {z})
        float3 xformOp:scale = ({sx}, {sy}, {sz})
        float xformOp:rotateY = {rot_y}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY", "xformOp:scale"]
        color3f[] primvars:displayColor = [({r}, {g}, {b})]
        custom string battery:cellType = "{cell_type}"
        custom string battery:cellId = "{cell_id}"
    }}
'''

_ROBOT_TEMPLATE = '''
    def Xform "RobotArm_{model}"
    {{
        float3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cylinder "Base"
        {{
            float3 xformOp:scale = (0.08, 0.08, 0.15)
            float3 xformOp:translate = (0, 0, 0.075)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.85, 0.20, 0.15)]
        }}

        custom string robot:model = "{model}"
        custom string robot:gripper = "{gripper}"
        custom double robot:maxReach = {max_reach}
        custom double robot:deadZone = {dead_zone}
        custom double robot:payload = {payload}
    }}
'''

_CAMERA_TEMPLATE = '''
    def Camera "InspectionCamera"
    {{
        float3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float focalLength = {focal}
        float horizontalAperture = 36
        custom float3 camera:lookAt = ({lx}, {ly}, {lz})
    }}
'''


# ── Cell color mapping ──
_CELL_COLORS = {
    "LG_E63":   (0.30, 0.69, 0.31),   # Green
    "HY_50Ah":  (0.13, 0.59, 0.95),   # Blue
    "CATL_LFP": (1.00, 0.60, 0.00),   # Orange
}
_DEFAULT_COLOR = (0.62, 0.62, 0.62)


def _build_cell_prim(cell: CellSpec) -> str:
    """Generate USDA for a single battery cell."""
    cat = CELL_CATALOG.get(cell.cell_type.value, {})
    w = cat.get("width", 0.05)
    d = cat.get("depth", 0.12)
    h = cat.get("height", 0.20)
    r, g, b = _CELL_COLORS.get(cell.cell_type.value, _DEFAULT_COLOR)

    return _CELL_TEMPLATE.format(
        cell_id=cell.id,
        x=cell.position[0],
        y=cell.position[1],
        z=cell.position[2] + h / 2,  # Lift half-height so bottom sits on tray
        sx=w / 2, sy=d / 2, sz=h / 2,
        rot_y=cell.rotation_y,
        r=r, g=g, b=b,
        cell_type=cell.cell_type.value,
    )


def _build_tray_prim(bounds: list[float]) -> str:
    """Generate USDA for the module tray."""
    tray_h = 0.02  # 2cm thick tray
    return _TRAY_TEMPLATE.format(
        cx=bounds[0] / 2,
        cy=bounds[1] / 2,
        cz=-tray_h / 2,
        sx=bounds[0] / 2,
        sy=bounds[1] / 2,
        sz=tray_h / 2,
        r=0.78, g=0.80, b=0.82,
    )


def _build_robot_prim(spec: ModuleTask) -> str:
    """Generate USDA for the robot arm."""
    robot = spec.robot
    profile = ROBOT_PROFILES.get(robot.model.value, ROBOT_PROFILES["UR10e"])
    return _ROBOT_TEMPLATE.format(
        model=robot.model.value,
        x=robot.base_position[0],
        y=robot.base_position[1],
        z=robot.base_position[2],
        gripper=robot.gripper.value,
        max_reach=profile["max_reach"],
        dead_zone=profile["dead_zone"],
        payload=profile["payload"],
    )


def _build_camera_prim(spec: ModuleTask) -> str:
    """Generate USDA for the inspection camera."""
    cam = spec.camera
    # Approximate focal length from FOV
    focal = 36.0 / (2.0 * __import__("math").tan(__import__("math").radians(cam.fov / 2)))
    return _CAMERA_TEMPLATE.format(
        x=cam.position[0],
        y=cam.position[1],
        z=cam.position[2],
        focal=round(focal, 2),
        lx=cam.look_at[0],
        ly=cam.look_at[1],
        lz=cam.look_at[2],
    )


def export_usda(spec: ModuleTask, output_path: str = "scene.usda") -> str:
    """
    Export a validated ModuleTask as a .usda scene file.

    Args:
        spec: Validated ModuleTask from the pipeline.
        output_path: Where to save the .usda file.

    Returns:
        Absolute path to the written .usda file.
    """
    children_parts = []

    # Lighting
    children_parts.append(_LIGHT_TEMPLATE)

    # Ground plane
    children_parts.append(_GROUND_TEMPLATE)

    # Module tray
    children_parts.append(_build_tray_prim(spec.module_tray_bounds))

    # All cells
    for cell in spec.cells:
        children_parts.append(_build_cell_prim(cell))

    # Robot arm
    children_parts.append(_build_robot_prim(spec))

    # Inspection camera
    children_parts.append(_build_camera_prim(spec))

    # Assemble
    usda_content = _USDA_HEADER.format(
        task_id=spec.task_id,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        children="".join(children_parts),
    )

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(usda_content, encoding="utf-8")

    return str(out.resolve())
