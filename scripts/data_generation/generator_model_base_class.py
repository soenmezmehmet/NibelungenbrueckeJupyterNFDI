import json
import gmsh
import dolfinx
from mpi4py import MPI

from utilities.checks import assert_path_exists
class GeneratorModel:
    ''' Base class for a generator of synthetic data from a model.'''

    def __init__(self, model_path:str, sensor_positions_path: str, model_parameters: dict):
        try:
            with open(sensor_positions_path) as f:
                self.sensor_positions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {sensor_positions_path} was not found")
        except json.decoder.JSONDecodeError:
            raise json.decoder.JSONDecodeError(f"The file {sensor_positions_path} is not a valid json")
        
        assert_path_exists(model_path)
        self.model_path = model_path

        self.model_parameters = model_parameters

    def Generate(self):
        ''' Generate the data from the start'''
        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()

    def LoadGeometry(self):
        ''' Load the meshed geometry from a .msh file'''
        
        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("mesh")

        # Import the .msh file
        gmsh.open(self.model_path)

        # Translate mesh from gmsh to dolfinx
        mesh = dolfinx.io.extract_gmsh_geometry(gmsh.model)
        self.mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, mesh.points, mesh.cells)
        
        gmsh.finalize()

    def GenerateModel(self):
        ''' Generate the FEM model.'''
        raise NotImplementedError("GenerateModel should be implemented")

    def GenerateData(self):
        ''' Run the FEM model and generate the data'''
        raise NotImplementedError("GenerateData should be implemented")