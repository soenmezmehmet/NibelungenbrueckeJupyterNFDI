import json
import dolfinx as df
import ufl
from mpi4py import MPI
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
import numpy as np
import pickle

class NullProblem:
    pass


class UpdateModelState:
    
    def __init__(self, dm, model_to_run, model_path):
        self.dm = dm
        self.model_params = {}
        self.model_to_run = model_to_run
        self.model_path = model_path
        self.mesh, cell_tags, facet_tags = df.io.gmshio.read_from_msh(self.model_path, MPI.COMM_WORLD, 0)
        self.dm.experiment = NullProblem()
        self.dm.problem = NullProblem()
        
            
    def reconstruct_model(self):
        if self.upload_model_parameters(self.model_to_run):
            self.restore_model()
        
        else:
            ##self.mesh, cell_tags, facet_tags = df.io.gmshio.read_from_msh(self.model_path, MPI.COMM_WORLD, 0)
            self.dm.experiment.mesh = self.mesh
            self.dm.solve()
            
    def upload_model_parameters(self, dm_path):
        try:
            with open(f"{dm_path}_params.pkl", 'rb') as f:
                self.model_params = pickle.load(f)
                print(f"Model loaded succesfully")
                self.model_params["displacement_values"] = np.array(self.model_params["displacement_values"])
                self.model_params["mesh_coordinates"] = np.array(self.model_params["mesh_coordinates"])
    
                return self.model_params
            
        except FileNotFoundError:
            print(f"Error: The file '{dm_path}' was not found!")    ##TODO: Use Assertion instead!!
            return None
        
        except Exception as e:
            print(f"An unexpected error!: {e}")
            
    def restore_model(self):
        displacement_values = self.model_params["displacement_values"]
        mesh_coordinates = self.model_params["mesh_coordinates"]

        #self.dm.experiment.mesh.geometry.x[:, :] = mesh_coordinates
        #self.dm.experiment.mesh = self.mesh
        
        degree = 2 
        dim = 3  
        #V = df.fem.VectorFunctionSpace(self.dm.experiment.mesh, ("Lagrange", degree))
        V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", degree))

        displacement_function = df.fem.Function(V)
        displacement_function.x.array[:] = displacement_values
        
        self.dm.problem.mesh = self.mesh
        self.dm.problem.V = V
        self.dm.problem.fields = SolutionFields(displacement=displacement_function)
        self.dm.solve() ##TODO:
        return self.dm
    
    def store_model_state(self):
        displacement_function = self.dm.problem.fields.displacement
        displacement_values = displacement_function.x.array[:]
        mesh_coordinates = self.dm.problem.mesh.geometry.x[:] 
        
        data_to_store = {
             "displacement_values": displacement_values.tolist(),
             "mesh_coordinates": mesh_coordinates.tolist(),
             "mesh_topology": str(self.dm.problem.mesh.topology.cell_type),
         
         }
        
        try:
            with open(f"{self.model_to_run}_params.pkl", "wb") as f:
                pickle.dump(data_to_store, f)
                print("Model state saved successfully.")
        except Exception as e:
            print(f"An error occurred while saving the model state: {e}")
            
            
    
            
            