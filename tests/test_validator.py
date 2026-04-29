import unittest
from schema import ModuleTask, CellSpec, RobotConfig, CellType, RobotModel, GripperType
from validator import validate, Severity

class TestValidator(unittest.TestCase):

    def setUp(self):
        from schema import InspectionCamera
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

if __name__ == '__main__':
    unittest.main()
