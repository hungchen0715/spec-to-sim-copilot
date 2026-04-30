"""
Industrial validation rules for battery module assembly specifications.

Each rule encodes a real physical or safety constraint from EV battery
manufacturing. These are NOT generic checks — they reflect actual
failure modes encountered in battery pack assembly lines:

1. Thermal Safety Gap   — cell chemistry mandates minimum clearance
2. Cell Alignment       — polarity misalignment breaks busbar welding
3. Robot Reachability   — arm work envelope from manufacturer specs
4. Module Tray Bounds   — cells must fit within the physical tray
5. Cell Count Limits    — practical limits for module assembly stations
"""
import math
from dataclasses import dataclass, field
from enum import Enum

from schema import ModuleTask, CellSpec
from config import (
    CELL_CATALOG,
    ROBOT_PROFILES,
    MAX_CELLS_PER_MODULE,
    MAX_CELLS_WARNING,
    CONVEYOR_CONSTRAINTS,
    PACKING_STATION_CONSTRAINTS,
)


class Severity(str, Enum):
    ERROR = "error"       # Assembly impossible — must fix
    WARNING = "warning"   # Possible issue — review recommended


@dataclass
class ValidationIssue:
    rule: str
    severity: Severity
    message: str
    fix_hint: str   # Actionable guidance for LLM repair loop


@dataclass
class ValidationReport:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _distance_3d(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 1: Thermal Safety Gap (Battery Cell Spacing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_cell_spacing(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: Lithium-ion cells generate heat during charge/discharge.
    If cells are packed too tightly, thermal runaway in one cell can cascade
    to neighbors. Min gap depends on cell chemistry:
    - LG_E63 (pouch): 2mm gap
    - HY_50Ah (prismatic): 3mm gap
    - CATL_LFP (prismatic): 2mm gap
    """
    issues = []
    cells = spec.cells

    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            cell_a = cells[i]
            cell_b = cells[j]

            dist = _distance_3d(cell_a.position, cell_b.position)

            # Get min gap from the stricter of the two cell types
            cat_a = CELL_CATALOG.get(cell_a.cell_type.value, {})
            cat_b = CELL_CATALOG.get(cell_b.cell_type.value, {})
            width_a = cat_a.get("width", 0.05)
            width_b = cat_b.get("width", 0.05)
            min_gap_a = cat_a.get("min_gap", 0.002)
            min_gap_b = cat_b.get("min_gap", 0.002)

            # Minimum center-to-center distance = half-width of each cell + gap
            min_center_dist = (width_a / 2) + (width_b / 2) + max(min_gap_a, min_gap_b)

            if dist < (width_a / 2 + width_b / 2) * 0.9:
                # Cells are overlapping
                issues.append(ValidationIssue(
                    rule="thermal_safety_gap",
                    severity=Severity.ERROR,
                    message=(
                        f"COLLISION: {cell_a.id} and {cell_b.id} are overlapping "
                        f"(distance: {dist*1000:.1f}mm, min required: {min_center_dist*1000:.1f}mm)."
                    ),
                    fix_hint=(
                        f"Move {cell_b.id} so its center is at least {min_center_dist*1000:.1f}mm "
                        f"from {cell_a.id}. Current distance is only {dist*1000:.1f}mm."
                    )
                ))
            elif dist < min_center_dist:
                issues.append(ValidationIssue(
                    rule="thermal_safety_gap",
                    severity=Severity.ERROR,
                    message=(
                        f"Safety Violation: {cell_a.id} and {cell_b.id} are {dist*1000:.1f}mm apart. "
                        f"Minimum thermal gap requires {min_center_dist*1000:.1f}mm center-to-center "
                        f"(cell widths: {width_a*1000:.0f}mm + {width_b*1000:.0f}mm + "
                        f"{max(min_gap_a, min_gap_b)*1000:.0f}mm gap)."
                    ),
                    fix_hint=(
                        f"Increase spacing between {cell_a.id} and {cell_b.id} to at least "
                        f"{min_center_dist*1000:.1f}mm center-to-center. This accounts for "
                        f"cell width + {max(min_gap_a, min_gap_b)*1000:.0f}mm thermal safety gap."
                    )
                ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 2: Cell Orientation / Polarity Alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_cell_alignment(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: In automated module assembly, all cells must have
    consistent polarity orientation for busbar welding. A misaligned cell
    means reversed polarity → short circuit → catastrophic failure.
    
    Allowed rotations: 0° or 180° (parallel alignment).
    Any other angle will break the automated welding jig.
    """
    issues = []
    cells = spec.cells

    if not cells:
        return issues

    reference_rotation = cells[0].rotation_y

    for cell in cells:
        # Normalize rotation to 0-360
        rot = cell.rotation_y % 360

        # Valid orientations: 0° or 180° (±0.5° tolerance for float precision)
        valid_orientations = [0.0, 180.0]
        is_valid = any(abs(rot - valid) < 0.5 for valid in valid_orientations)

        if not is_valid:
            issues.append(ValidationIssue(
                rule="cell_alignment",
                severity=Severity.ERROR,
                message=(
                    f"ORIENTATION ERROR: {cell.id} rotation is {cell.rotation_y}°. "
                    f"Busbar welding requires 0° or 180° alignment only."
                ),
                fix_hint=(
                    f"Set {cell.id}'s rotation_y to either 0.0 or 180.0 degrees. "
                    f"Current value {cell.rotation_y}° will cause busbar welding failure."
                )
            ))

    # Check consistency: all should be same OR alternating 0/180
    rotations = [c.rotation_y % 360 for c in cells]
    unique_rots = set(round(r) for r in rotations)
    if len(unique_rots) > 2:
        issues.append(ValidationIssue(
            rule="cell_alignment",
            severity=Severity.WARNING,
            message=(
                f"Inconsistent cell orientations detected: {unique_rots}. "
                f"Standard modules use either uniform or alternating 0°/180° patterns."
            ),
            fix_hint=(
                "Standardize all cell rotations. Use 0.0 for all cells (uniform polarity) "
                "or alternate between 0.0 and 180.0 for paired configurations."
            )
        ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 3: Robot Arm Reachability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_robot_reach(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: Every robot arm has a physical work envelope defined
    by its max reach radius. If any cell placement falls outside this
    envelope, the robot physically cannot reach it → simulation crash
    or unrealistic motion planning.
    
    Also checks the dead zone (area too close to robot base where the
    arm cannot fold back enough to operate).
    """
    issues = []
    robot = spec.robot
    profile = ROBOT_PROFILES.get(robot.model.value, ROBOT_PROFILES["UR10e"])
    max_reach = profile["max_reach"]
    dead_zone = profile["dead_zone"]
    base_pos = robot.base_position

    for cell in spec.cells:
        dist = _distance_3d(base_pos, cell.position)

        if dist > max_reach:
            issues.append(ValidationIssue(
                rule="robot_reachability",
                severity=Severity.ERROR,
                message=(
                    f"OUT OF REACH: {cell.id} at [{cell.position[0]:.3f}, "
                    f"{cell.position[1]:.3f}, {cell.position[2]:.3f}] is "
                    f"{dist:.3f}m from robot base. "
                    f"{robot.model.value} max reach is {max_reach}m."
                ),
                fix_hint=(
                    f"Move {cell.id} closer to the robot base at "
                    f"{base_pos}. Maximum distance is {max_reach}m. "
                    f"Current distance: {dist:.3f}m. "
                    f"Consider repositioning the robot base or choosing a longer-reach robot."
                )
            ))
        elif dist < dead_zone:
            issues.append(ValidationIssue(
                rule="robot_reachability",
                severity=Severity.ERROR,
                message=(
                    f"DEAD ZONE: {cell.id} is only {dist:.3f}m from robot base. "
                    f"{robot.model.value} cannot operate within {dead_zone}m of its base."
                ),
                fix_hint=(
                    f"Move {cell.id} at least {dead_zone}m away from the robot base. "
                    f"The dead zone is the area where the arm cannot physically fold to reach."
                )
            ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 4: Module Tray Bounds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_tray_bounds(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: The module tray is a physical fixture with fixed
    dimensions. Any cell placed outside the tray will fall off the
    assembly line or collide with adjacent stations.
    """
    issues = []
    bounds = spec.module_tray_bounds

    for cell in spec.cells:
        cat = CELL_CATALOG.get(cell.cell_type.value, {})
        half_w = cat.get("width", 0.05) / 2
        half_d = cat.get("depth", 0.12) / 2

        # Check X bounds
        if cell.position[0] - half_w < 0 or cell.position[0] + half_w > bounds[0]:
            issues.append(ValidationIssue(
                rule="tray_bounds",
                severity=Severity.ERROR,
                message=(
                    f"{cell.id} extends outside module tray on X-axis. "
                    f"Cell at x={cell.position[0]:.3f}m, tray width={bounds[0]}m."
                ),
                fix_hint=(
                    f"Move {cell.id} so its x-position is between {half_w:.3f} "
                    f"and {bounds[0] - half_w:.3f}."
                )
            ))

        # Check Y bounds
        if cell.position[1] - half_d < 0 or cell.position[1] + half_d > bounds[1]:
            issues.append(ValidationIssue(
                rule="tray_bounds",
                severity=Severity.ERROR,
                message=(
                    f"{cell.id} extends outside module tray on Y-axis. "
                    f"Cell at y={cell.position[1]:.3f}m, tray depth={bounds[1]}m."
                ),
                fix_hint=(
                    f"Move {cell.id} so its y-position is between {half_d:.3f} "
                    f"and {bounds[1] - half_d:.3f}."
                )
            ))

        # Check Z (should not be negative)
        if cell.position[2] < 0:
            issues.append(ValidationIssue(
                rule="tray_bounds",
                severity=Severity.ERROR,
                message=f"{cell.id} z-position is {cell.position[2]:.3f}m — below tray surface.",
                fix_hint=f"Set {cell.id} z-position to 0.0 (tray surface level)."
            ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 5: Cell Count Limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_cell_count(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: Assembly stations have practical limits on how many
    cells can be placed in a single module. Exceeding this creates
    cycle-time bottlenecks and thermal management challenges.
    """
    issues = []
    count = len(spec.cells)

    if count > MAX_CELLS_PER_MODULE:
        issues.append(ValidationIssue(
            rule="cell_count",
            severity=Severity.ERROR,
            message=(
                f"Module has {count} cells — exceeds station limit of "
                f"{MAX_CELLS_PER_MODULE}."
            ),
            fix_hint=(
                f"Reduce cell count to at most {MAX_CELLS_PER_MODULE}. "
                f"Split into multiple modules if more cells are needed."
            )
        ))
    elif count > MAX_CELLS_WARNING:
        issues.append(ValidationIssue(
            rule="cell_count",
            severity=Severity.WARNING,
            message=(
                f"Module has {count} cells — approaching station limit "
                f"({MAX_CELLS_WARNING} recommended max)."
            ),
            fix_hint=(
                f"Consider reducing to {MAX_CELLS_WARNING} cells or fewer "
                f"for optimal assembly cycle time."
            )
        ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 6: Conveyor Belt Collision Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_conveyor_collision(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: Conveyor belts on a factory floor cannot overlap
    or intersect each other. If two belts occupy the same space, physical
    modules will collide during transport.

    Also validates belt length against practical factory constraints.
    """
    issues = []
    conveyors = spec.conveyors
    if not conveyors:
        return issues

    margin = CONVEYOR_CONSTRAINTS["overlap_margin"]
    min_len = CONVEYOR_CONSTRAINTS["min_length"]
    max_len = CONVEYOR_CONSTRAINTS["max_length"]

    for conv in conveyors:
        # Check belt length
        belt_length = _distance_3d(conv.start_position, conv.end_position)
        if belt_length < min_len:
            issues.append(ValidationIssue(
                rule="conveyor_collision",
                severity=Severity.ERROR,
                message=(
                    f"Conveyor {conv.id} is only {belt_length:.3f}m long. "
                    f"Minimum belt length is {min_len}m."
                ),
                fix_hint=(
                    f"Move the start or end position of {conv.id} so the belt "
                    f"is at least {min_len}m long."
                )
            ))
        elif belt_length > max_len:
            issues.append(ValidationIssue(
                rule="conveyor_collision",
                severity=Severity.WARNING,
                message=(
                    f"Conveyor {conv.id} is {belt_length:.3f}m long — exceeds "
                    f"typical factory limit of {max_len}m."
                ),
                fix_hint=(
                    f"Consider splitting {conv.id} into two shorter segments "
                    f"with a transfer station."
                )
            ))

    # Check pairwise overlap (axis-aligned bounding box)
    for i in range(len(conveyors)):
        for j in range(i + 1, len(conveyors)):
            c_a = conveyors[i]
            c_b = conveyors[j]

            # Compute midpoint distance as simple collision proxy
            mid_a = [(s + e) / 2 for s, e in zip(c_a.start_position, c_a.end_position)]
            mid_b = [(s + e) / 2 for s, e in zip(c_b.start_position, c_b.end_position)]
            dist = _distance_3d(mid_a, mid_b)

            # Min safe distance = half-widths + margin
            min_safe = (c_a.width / 2) + (c_b.width / 2) + margin

            if dist < min_safe:
                issues.append(ValidationIssue(
                    rule="conveyor_collision",
                    severity=Severity.ERROR,
                    message=(
                        f"Conveyors {c_a.id} and {c_b.id} are too close "
                        f"(midpoint distance: {dist:.3f}m, min required: {min_safe:.3f}m)."
                    ),
                    fix_hint=(
                        f"Move {c_b.id} at least {min_safe:.3f}m away from {c_a.id} "
                        f"(accounting for belt widths + {margin*1000:.0f}mm safety margin)."
                    )
                ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 7: Packing Station Robot Reachability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_packing_reach(spec: ModuleTask) -> list[ValidationIssue]:
    """
    Domain rationale: A packing station's robot arm must be able to
    physically reach the end of its connected conveyor belt to pick
    up modules. If the station is too far from the conveyor endpoint,
    the packing operation cannot execute.

    Also validates that the referenced conveyor_in ID actually exists.
    """
    issues = []
    stations = spec.packing_stations
    conveyors = spec.conveyors

    if not stations:
        return issues

    # Build conveyor lookup
    conveyor_map = {}
    if conveyors:
        conveyor_map = {c.id: c for c in conveyors}

    max_reach = PACKING_STATION_CONSTRAINTS["max_reach_to_conveyor"]

    for station in stations:
        # Check if referenced conveyor exists
        if station.conveyor_in not in conveyor_map:
            issues.append(ValidationIssue(
                rule="packing_reach",
                severity=Severity.ERROR,
                message=(
                    f"Packing station {station.id} references conveyor "
                    f"'{station.conveyor_in}' which does not exist."
                ),
                fix_hint=(
                    f"Add a ConveyorBelt with id='{station.conveyor_in}' to the spec, "
                    f"or update {station.id}'s conveyor_in to an existing belt ID."
                )
            ))
            continue

        # Check reach to conveyor endpoint
        conveyor = conveyor_map[station.conveyor_in]
        dist_to_end = _distance_3d(station.position, conveyor.end_position)

        # Also get robot-specific reach limit
        robot_profile = ROBOT_PROFILES.get(
            station.robot.model.value, ROBOT_PROFILES["UR10e"]
        )
        effective_reach = min(max_reach, robot_profile["max_reach"])

        if dist_to_end > effective_reach:
            issues.append(ValidationIssue(
                rule="packing_reach",
                severity=Severity.ERROR,
                message=(
                    f"Packing station {station.id} is {dist_to_end:.3f}m from "
                    f"conveyor {conveyor.id} endpoint. "
                    f"{station.robot.model.value} max reach is {effective_reach}m."
                ),
                fix_hint=(
                    f"Move {station.id} closer to the end of conveyor {conveyor.id}, "
                    f"or use a robot with longer reach. Max distance: {effective_reach}m."
                )
            ))

    # Check packing station clearance
    min_clear = PACKING_STATION_CONSTRAINTS["min_clearance"]
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            dist = _distance_3d(stations[i].position, stations[j].position)
            if dist < min_clear:
                issues.append(ValidationIssue(
                    rule="packing_reach",
                    severity=Severity.WARNING,
                    message=(
                        f"Packing stations {stations[i].id} and {stations[j].id} "
                        f"are only {dist:.3f}m apart (min: {min_clear}m)."
                    ),
                    fix_hint=(
                        f"Space packing stations at least {min_clear}m apart "
                        f"to avoid robot arm collisions."
                    )
                ))

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main validation entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def validate(spec: ModuleTask) -> ValidationReport:
    """Run all industrial validation rules and produce a structured report."""
    all_issues: list[ValidationIssue] = []

    all_issues.extend(_check_cell_spacing(spec))
    all_issues.extend(_check_cell_alignment(spec))
    all_issues.extend(_check_robot_reach(spec))
    all_issues.extend(_check_tray_bounds(spec))
    all_issues.extend(_check_cell_count(spec))
    all_issues.extend(_check_conveyor_collision(spec))
    all_issues.extend(_check_packing_reach(spec))

    has_errors = any(i.severity == Severity.ERROR for i in all_issues)

    # Calculate assembly stats
    total_weight = sum(
        CELL_CATALOG.get(c.cell_type.value, {}).get("weight", 0.0)
        for c in spec.cells
    )

    stats = {
        "cell_count": len(spec.cells),
        "cell_types": list(set(c.cell_type.value for c in spec.cells)),
        "total_weight": f"{total_weight:.2f} kg",
        "robot_model": spec.robot.model.value,
        "gripper": spec.robot.gripper.value,
        "tray_size": f"{spec.module_tray_bounds[0]}m × {spec.module_tray_bounds[1]}m",
    }

    return ValidationReport(
        passed=not has_errors,
        issues=all_issues,
        stats=stats,
    )


def format_report(report: ValidationReport) -> str:
    """Format a validation report as human-readable text."""
    lines = []

    if report.passed:
        lines.append("✅ VALIDATION PASSED — Module spec is ready for USD generation")
    else:
        error_count = sum(1 for i in report.issues if i.severity == Severity.ERROR)
        warn_count = sum(1 for i in report.issues if i.severity == Severity.WARNING)
        lines.append(
            f"❌ VALIDATION FAILED — {error_count} error(s), {warn_count} warning(s)"
        )

    lines.append("")

    if report.issues:
        lines.append("Issues:")
        for issue in report.issues:
            icon = "🔴" if issue.severity == Severity.ERROR else "🟡"
            lines.append(f"  {icon} [{issue.rule}] {issue.message}")
    else:
        lines.append("No issues found.")

    lines.append("")
    lines.append("Assembly Stats:")
    for key, value in report.stats.items():
        lines.append(f"  • {key}: {value}")

    return "\n".join(lines)


def format_issues_for_llm(report: ValidationReport) -> str:
    """Format issues with fix_hints for the LLM repair prompt."""
    lines = []
    for issue in report.issues:
        lines.append(
            f"[{issue.severity.value.upper()}] {issue.rule}: {issue.message}\n"
            f"  Fix hint: {issue.fix_hint}"
        )
    return "\n\n".join(lines)
