import unittest
import tempfile
import shutil
from nibelungenbruecke.scripts.digital_twin_orchestrator.base_model import BaseModel
import dolfinx
from mpi4py import MPI

# Create a dummy subclass of BaseModel for testing
class DummyModel(BaseModel):
    def GenerateModel(self):
        pass

    def GenerateData(self):
        pass

    def update_input(self, sensor_input):
        pass

    def solve(self):
        pass

    def export_output(self):
        pass

    @staticmethod
    def _get_default_parameters():
        return {"param1": 0, "param2": 0}


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.test_model_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh"
        self.test_model_parameters = {"param1": 1, "param2": 2}



    # @classmethod
    # def setUpClass(cls):
    #     # Create a temporary directory for testing
    #     cls.temp_dir = tempfile.mkdtemp()
    #     cls.temp_model_path = f"{cls.temp_dir}/test_model.msh"
    #     cls.temp_model_parameters = {"param1": 1, "param2": 2}

    # @classmethod
    # def tearDownClass(cls):
    #     # Cleanup temporary directory
    #     shutil.rmtree(cls.temp_dir)

    def test_initialization(self):
        """Test BaseModel initialization."""
        base_model = BaseModel(self.test_model_path, self.test_model_parameters)
        self.assertEqual(base_model.model_path, self.test_model_path)
        self.assertEqual(base_model.model_parameters, self.test_model_parameters)

    def test_load_geometry(self):
        """Test LoadGeometry method."""
        dummy_model = DummyModel(self.test_model_path, self.test_model_parameters)
        dummy_model.LoadGeometry()
        self.assertIsInstance(dummy_model.mesh, dolfinx.mesh.Mesh)

    def test_generate(self):
        """Test Generate method."""
        dummy_model = DummyModel(self.test_model_path, self.test_model_parameters)
        # Assuming Generate should not raise any exception
        try:
            dummy_model.Generate()
        except Exception as e:
            self.fail(f"Generate method raised an exception: {e}")

    def test_validate_parameters(self):
        """Test _validate_parameters method."""
        base_model = BaseModel(self.test_model_path, self.test_model_parameters)
        self.assertEqual(base_model.model_parameters, {"param1": 1, "param2": 2})


if __name__ == "__main__":
    unittest.main()
