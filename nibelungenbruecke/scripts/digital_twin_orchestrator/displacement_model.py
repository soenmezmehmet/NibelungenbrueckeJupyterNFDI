from nibelungenbruecke.scripts.digital_twin_orchestrator.base_model import BaseModel
#from base_model import BaseModel
import dolfinx as df
import json
from nibelungenbruecke.scripts.data_generation.nibelungen_experiment import NibelungenExperiment
from fenicsxconcrete.finite_element_problem.linear_elasticity_nibelungenbruecke_demonstrator import LinearElasticityNibelungenbrueckeDemonstrator
from fenicsxconcrete.util import ureg
from nibelungenbruecke.scripts.utilities.API_sensor_retrieval import API_Request, MetadataSaver, Translator
from nibelungenbruecke.scripts.utilities.loaders import load_sensors
from nibelungenbruecke.scripts.utilities.offloaders import offload_sensors
import importlib
import time

class DisplacementModel(BaseModel):
    
    def __init__(self, model_path: str, model_parameters: dict, dt_path: str):
        super().__init__(model_path, model_parameters)
        
        self.model_parameters = model_parameters
        self.material_parameters = self.model_parameters["material_parameters"]
        self.default_p = self._get_default_parameters()
        self.dt_path = dt_path
        self.vs_path = self.model_parameters["virtual_sensor_added_output_path"] ##TODO: !!
        #self.experiment = NibelungenExperiment(self.model_path, self.model_parameters)
        
    def LoadGeometry(self):
        pass
    
    def GenerateModel(self):
        
        self.experiment = NibelungenExperiment(self.model_path, self.model_parameters)
        #self.experiment = NibelungenExperiment(self.model_path, self.material_parameters)
        self.default_p.update(self.experiment.default_parameters()) ## TODO: self.default_p.update(self.experiment.parameters)
        self.problem = LinearElasticityNibelungenbrueckeDemonstrator(self.experiment, self.default_p)
        
    def GenerateData(self):
        """Generate data based on the model parameters."""

        self.api_request = API_Request()
        self.api_dataFrame = self.api_request.fetch_data()

        metadata_saver = MetadataSaver(self.model_parameters, self.api_dataFrame)
        metadata_saver.saving_metadata()

        translator = Translator(self.model_parameters)
        translator.translator_to_sensor()

        self.problem.import_sensors_from_metadata(self.model_parameters["MKP_meta_output_path"])
        self.problem.fields.temperature = self.problem.fields.displacement #!!
        self.problem.dynamic_solve()        ##TODO: change the name!

        translator.save_to_MKP(self.api_dataFrame)
        translator.save_virtual_sensor(self.problem)

        if self.model_parameters["paraview_output"]:
            with df.io.XDMFFile(self.problem.mesh.comm, self.model_parameters["paraview_output_path"]+"/"+self.model_parameters["model_name"]+".xdmf", "w") as xdmf:
                xdmf.write_mesh(self.problem.mesh)
                xdmf.write_function(self.problem.fields.displacement)
        
    @staticmethod
    def _get_default_parameters():
        """
        Get default material parameters.

        Returns:
            dict: Default material parameters.
        """
        default_parameters = {
            "rho":7750 * ureg("kg/m^3"),
            "E":210e9 * ureg("N/m^2"),
            "nu":0.28 * ureg("")
        }
        return default_parameters
    
    def update_input(self, sensor_input):
        
        with open(self.dt_path, 'r') as f:
            dt_params = json.load(f)
        
        # currently, only updates E value
        #TODO: Make this part more automated/flexible!  
        if isinstance(sensor_input, (int, float)):
            dt_params[0]["parameters"]["E"] = sensor_input    ##TODO: Change to rho!!
            
            with open(self.dt_path, 'w') as file:
                json.dump(dt_params, file, indent=4)
            return True
        else:
            return False
        
    def solve(self):

        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()
        
        #TODO: API Request output error!!
        self.sensor_out = self.api_dataFrame['E_plus_080DU_HSN-u-_Avg1'].iloc[-1] # *1000 #Convertion from meter to milimeter
        #self.sensor_out = self.api_dataFrame['E_plus_413TU_HSS-m-_Avg1'].iloc[-1]
   
        
        vs_file_path = self.model_parameters["virtual_sensor_added_output_path"]
        with open(vs_file_path, 'r') as file:
            self.vs_data = json.load(file)        
        
        #TODO: API Request output error!!
        #self.vs_sensor_out = self.vs_data['virtual_sensors']['E_plus_413TU_HSS-m-_Avg1']['displacements'][-1][0]
        self.vs_sensor_out = self.vs_data['virtual_sensors']['E_plus_080DU_HSN-u-_Avg1']['displacements'][-1][0]
        
    def export_output(self, path: str): #TODO: json_path as a input parameters!! -> Changes' been done!
        #json_path = "output_data.json" #TODO: move to json file
        
        json_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" + path + ".json"
        try:
            with open(json_path, 'r') as file:
                output_data = json.load(file)
                
        except FileNotFoundError:
            output_data = {}
            
        output_data.setdefault('real_sensor_output', []).append(self.sensor_out)
        output_data.setdefault('virtual_sensor_output', []).append(self.vs_sensor_out)

        local_time = time.localtime()
        output_data.setdefault('time', []).append(time.strftime("%y-%m-%d %H:%M:%S", local_time))

        with open(self.dt_path, 'r') as f:
            dt_params = json.load(f)
        output_data.setdefault('Input_parameter', []).append(dt_params[0]["parameters"]["E"])

        
        with open(json_path, 'w') as file:
            json.dump(output_data, file)
            
        return json_path
 
    
if __name__ == "__main__": 
    
    model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
    
    model_parameters = {'model_name': 'displacements',
     'df_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_df_output.csv',
     'meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_meta_output.json',
     'MKP_meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_meta_output.json',
     'MKP_translated_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_translated.json',
     'virtual_sensor_added_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/virtual_sensor_added_translated.json',
     'cache_path': '',
     'paraview_output': True,
     'paraview_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview',
     'material_parameters': {'E': 40000000000000.0, 'nu': 0.2, 'rho': 2350},
     'tension_z': 0.0,
     'mass': 50000.0,
     'g': 9.81,
     'initial_position': [0.0, 0.0, 0.0],
     'speed': 1.0,
     'length': 7.5,
     'width': 2.5,
     'height': 6.5,
     'length_road': 95.0,
     'width_road': 14.0,
     'thickness_deck': 0.2,
     'dt': 30.0,
     'reference_temperature': 300,
     'temperature_coefficient': 1e-05,
     'temperature_alpha': 1e-05,
     'temperature_difference': 5.0,
     'reference_height': -2.5,
     'boundary_conditions': {'bc1': {'model': 'clamped_edge',
       'side_coord_1': 0.0,
       'coord_1': 2,
       'side_coord_2': 0.0,
       'coord_2': 1},
      'bc2': {'model': 'clamped_edge',
       'side_coord_1': 95.185,
       'coord_1': 2,
       'side_coord_2': 0.0,
       'coord_2': 1}}}
    
    dt_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_parameters.json'
    
    
    dm = DisplacementModel(model_path, model_parameters, dt_path)
    dm.solve()

#%%
# =============================================================================
# import pickle
# import numpy as np
# 
# displacement_function = dm.problem.fields.displacement
# displacement_values = displacement_function.x.array[:]
# mesh_coordinates = dm.problem.mesh.geometry.x[:] 
# 
# data_to_store = {
#     "displacement_values": displacement_values,
#     "mesh_coordinates": mesh_coordinates,
#     "mesh_topology": dm.problem.mesh.topology.cell_type,
# 
# }
# 
# with open("test.pkl", "wb") as f:
#     pickle.dump(data_to_store, f)
# 
# print("Deflected model saved successfully.")
# 
# #%% Upload part!!
# with open("test.pkl", "rb") as f:
#     stored_data = pickle.load(f)
# 
# # Access stored values
# displacement_values = stored_data["displacement_values"]
# mesh_coordinates = stored_data["mesh_coordinates"]
# 
# #%%
# 
# import pickle
# import numpy as np
# import dolfinx as df
# import ufl
# from mpi4py import MPI
# from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
# 
# 
# with open("test.pkl", "rb") as f:
#     stored_data = pickle.load(f)
# 
# 
# displacement_values = stored_data["displacement_values"]
# mesh_coordinates = stored_data["mesh_coordinates"]
# 
# 
# mesh_file_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh"  # Replace with your actual mesh file path
# mesh, cell_tags, facet_tags = df.io.gmshio.read_from_msh(mesh_file_path, MPI.COMM_WORLD, 0)
# 
# 
# mesh.geometry.x[:, :] = mesh_coordinates
# 
# degree = 2 
# dim = 3  
# V = df.fem.VectorFunctionSpace(mesh, ("Lagrange", degree))
# 
# 
# displacement_function = df.fem.Function(V)
# displacement_function.x.array[:] = displacement_values
# 
# dm_1 = DisplacementModel(model_path, model_parameters, dt_path)
# dm_1.GenerateModel()
# 
# dm_1.problem.V = V
# dm_1.problem.fields = SolutionFields(displacement=displacement_function)
# 
# dm_1.solve()
# 
# dm_1.problem.sensors.get("E_plus_080DU_HSN-o-_Avg1", None).data[0].tolist()
#    
# 
# class NullProblem:
#     pass
# 
# dm_2 = dm_1
# dm_2.problem = NullProblem()
# 
# dm_2.problem.V = V
# dm_2.problem.fields = SolutionFields(displacement=displacement_function)
# 
# dm_2.solve()
# 
# dm_2.problem.sensors.get("E_plus_080DU_HSN-o-_Avg1", None).data[0].tolist()
#    
# 
# =============================================================================
