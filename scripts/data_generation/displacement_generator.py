import json

import dolfinx
import gmsh
from mpi4py import MPI

from generator_model_base_class import GeneratorModel

class DisplacementGenerator(GeneratorModel):
    ''' Generates the displacements at the sensors for a given load configuration.'''
    
    def __init__(self, model_path: str, sensor_positions_path: str, model_parameters: dict):
        super().__init__(model_path, sensor_positions_path, model_parameters)


    def LoadGeometry(self):
        # Code to load the displacement model

        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("mesh")
        # Import the .geo_unrolled file
        gmsh.open(self.model_path)

        # Translate mesh from gmsh to dolfinx
        mesh = dolfinx.io.extract_gmsh_geometry(gmsh.model)
        self.mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, mesh.points, mesh.cells)

    def GenerateModel(self):
        # Code to generate the displacement model
        pass

    def GenerateData(self):
        # Code to generate displacement data
        pass

    def _get_default_parameters():
        default_parameters = {
            "model_path": "input/models/mesh.msh",
            "output_path": "data",
            "output_format": ".h5",
            "generation_models_list": [None],
        }

        return default_parameters
