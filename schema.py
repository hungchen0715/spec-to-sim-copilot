"""
Pydantic data models for the battery factory AI-to-USD pipeline.

These schemas define the structured output that the LLM must produce
for generating validated battery module assembly scenes in Isaac Sim.
Every field maps to real EV battery manufacturing constraints:
- Cell dimensions from LG / Hyundai / CATL datasheets
- Thermal safety gaps mandated by cell chemistry
- Robot work envelope from UR / KUKA / FANUC specs
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Battery cell types (real manufacturer SKUs) ──
class CellType(str, Enum):
    LG_E63 = "LG_E63"        # LG pouch cell, 63Ah
    HY_50Ah = "HY_50Ah"      # Hyundai prismatic, 50Ah
    CATL_LFP = "CATL_LFP"    # CATL LFP prismatic, 161Ah


# ── Robot model selection ──
class RobotModel(str, Enum):
    UR10e = "UR10e"
    KUKA_KR6 = "KUKA_KR6"
    FANUC_CRX10 = "FANUC_CRX10"


# ── Gripper types ──
class GripperType(str, Enum):
    VACUUM = "Vacuum_Gripper_V1"
    CLAMP = "Mechanical_Clamp_V2"
    SOFT = "Soft_Finger_V1"


# ── A single battery cell in the module ──
class CellSpec(BaseModel):
    id: str = Field(
        ...,
        description="Unique cell identifier, e.g. 'Cell_01'"
    )
    cell_type: CellType = Field(
        ...,
        description="Battery cell model from catalogue"
    )
    position: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="[x, y, z] placement position in meters, relative to module tray origin"
    )
    rotation_y: float = Field(
        default=0.0,
        description="Rotation around Y-axis in degrees (for polarity alignment)"
    )


# ── Robot arm configuration ──
class RobotConfig(BaseModel):
    model: RobotModel = Field(
        default=RobotModel.UR10e,
        description="Robot arm model"
    )
    base_position: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Robot base mounting position [x, y, z] in meters"
    )
    gripper: GripperType = Field(
        default=GripperType.VACUUM,
        description="End-effector gripper type"
    )


# ── Camera for inspection / monitoring ──
class InspectionCamera(BaseModel):
    position: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Camera position [x, y, z] in meters"
    )
    look_at: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Camera target point [x, y, z]"
    )
    fov: float = Field(
        default=60.0,
        ge=10.0,
        le=120.0,
        description="Field of view in degrees"
    )


# ── Top-level: Complete Module Assembly Task ──
class ModuleTask(BaseModel):
    task_id: str = Field(
        ...,
        description="Unique task identifier, e.g. 'BatteryModule_2x3_LG'"
    )
    description: str = Field(
        ...,
        description="One-line summary of the assembly task"
    )
    cells: list[CellSpec] = Field(
        ...,
        min_length=1,
        description="All battery cells to be placed in the module"
    )
    robot: RobotConfig = Field(
        default_factory=RobotConfig,
        description="Robot arm configuration for this assembly task"
    )
    camera: InspectionCamera = Field(
        ...,
        description="Inspection camera setup"
    )
    module_tray_bounds: list[float] = Field(
        default=[0.8, 0.6, 0.3],
        min_length=3,
        max_length=3,
        description="Module tray dimensions [x, y, z] in meters"
    )
