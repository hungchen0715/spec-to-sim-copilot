import unittest
from schema import (
    ModuleTask, CellSpec, RobotConfig, CellType, RobotModel,
    GripperType, InspectionCamera, ConveyorBelt, PackingStation,
)
from validator import validate, Severity

class TestValidator(unittest.TestCase):

    def setUp(self):
        self.base_spec = ModuleTask(
            task_id="Test_Task_01",
            description="Unit test task",
            cells=[
                CellSpec(id="Cell_01", cell_type=CellType.LG_E63, position=[0.2, 0.2, 0.0], rotation_y=0.0),
                CellSpec(id="Cell_02", cell_type=CellType.LG_E63, position=[0.3, 0.2, 0.0], rotation_y=180.0),
            ],
            robot=RobotConfig(model=RobotModel.UR10e, gripper=GripperType.VACUUM, base_position=[0.0, 0.0, 0.0]),
            camera=InspectionCamera(position=[0.0, 0.0, 1.5], look_at=[0.0, 0.0, 0.0]),
            module_tray_bounds=[1.0, 1.0, 0.1]
        )

    def test_valid_spec(self):
        report = validate(self.base_spec)
        self.assertTrue(report.passed)

    def test_thermal_safety_gap(self):
        # Move cells too close to each other (40mm apart in X)
        self.base_spec.cells[1].position = [0.24, 0.2, 0.0] 
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.rule == "thermal_safety_gap" and i.severity == Severity.ERROR for i in report.issues))

    def test_robot_reachability(self):
        # Move cell out of reach for UR10e (max reach 1.3m)
        self.base_spec.cells[0].position = [2.0, 2.0, 0.0]
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.rule == "robot_reachability" for i in report.issues))

    # ── Conveyor Belt Tests ──

    def test_valid_conveyor_spec(self):
        """A valid conveyor + packing station should pass all rules."""
        self.base_spec.conveyors = [
            ConveyorBelt(
                id="Conv_01",
                start_position=[0.0, 0.0, 0.0],
                end_position=[2.0, 0.0, 0.0],
                speed_mps=0.5,
                width=0.6,
            ),
        ]
        self.base_spec.packing_stations = [
            PackingStation(
                id="Pack_01",
                position=[2.0, 0.5, 0.0],
                conveyor_in="Conv_01",
            ),
        ]
        report = validate(self.base_spec)
        self.assertTrue(report.passed)

    def test_conveyor_too_short(self):
        """A conveyor belt shorter than 0.5m should trigger an error."""
        self.base_spec.conveyors = [
            ConveyorBelt(
                id="Conv_Short",
                start_position=[0.0, 0.0, 0.0],
                end_position=[0.1, 0.0, 0.0],  # Only 0.1m
                width=0.6,
            ),
        ]
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(
            i.rule == "conveyor_collision" and i.severity == Severity.ERROR
            for i in report.issues
        ))

    def test_conveyors_overlap(self):
        """Two conveyors placed too close should trigger collision error."""
        self.base_spec.conveyors = [
            ConveyorBelt(
                id="Conv_A",
                start_position=[0.0, 0.0, 0.0],
                end_position=[2.0, 0.0, 0.0],
                width=0.6,
            ),
            ConveyorBelt(
                id="Conv_B",
                start_position=[0.0, 0.1, 0.0],  # Too close (0.1m apart, widths 0.6m each)
                end_position=[2.0, 0.1, 0.0],
                width=0.6,
            ),
        ]
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(
            i.rule == "conveyor_collision" and "too close" in i.message
            for i in report.issues
        ))

    # ── Packing Station Tests ──

    def test_packing_station_too_far(self):
        """Packing station out of robot reach from conveyor endpoint."""
        self.base_spec.conveyors = [
            ConveyorBelt(
                id="Conv_01",
                start_position=[0.0, 0.0, 0.0],
                end_position=[2.0, 0.0, 0.0],
                width=0.6,
            ),
        ]
        self.base_spec.packing_stations = [
            PackingStation(
                id="Pack_Far",
                position=[5.0, 5.0, 0.0],  # Way too far
                conveyor_in="Conv_01",
            ),
        ]
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.rule == "packing_reach" for i in report.issues))

    def test_packing_station_missing_conveyor(self):
        """Packing station referencing a non-existent conveyor ID."""
        self.base_spec.packing_stations = [
            PackingStation(
                id="Pack_01",
                position=[1.0, 0.5, 0.0],
                conveyor_in="Conv_MISSING",  # Doesn't exist
            ),
        ]
        report = validate(self.base_spec)
        self.assertFalse(report.passed)
        self.assertTrue(any(
            i.rule == "packing_reach" and "does not exist" in i.message
            for i in report.issues
        ))

if __name__ == '__main__':
    unittest.main()

