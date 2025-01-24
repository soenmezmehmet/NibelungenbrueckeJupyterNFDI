import json
import importlib
import time
import pickle

import dolfinx as df
from fenicsxconcrete.util import ureg
from fenicsxconcrete.finite_element_problem.linear_elasticity_nibelungenbruecke_demonstrator import LinearElasticityNibelungenbrueckeDemonstrator

from nibelungenbruecke.scripts.utilities.loaders import load_sensors
from nibelungenbruecke.scripts.utilities.offloaders import offload_sensors
from nibelungenbruecke.scripts.digital_twin_orchestrator.base_model import BaseModel
from nibelungenbruecke.scripts.data_generation.nibelungen_experiment import NibelungenExperiment
from nibelungenbruecke.scripts.utilities.API_sensor_retrieval import API_Request, MetadataSaver, Translator

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
        self.default_p.update(self.experiment.default_parameters()) ## TODO: self.default_p.update(self.experiment.parameters)
        self.problem = LinearElasticityNibelungenbrueckeDemonstrator(self.experiment, self.default_p)
        
    def GenerateData(self):
        """Generate data based on the model parameters."""

        self.api_request = API_Request(self.model_parameters["secret_path"])
        self.api_dataFrame = self.api_request.fetch_data()

        metadata_saver = MetadataSaver(self.model_parameters, self.api_dataFrame)
        metadata_saver.saving_metadata()

        translator = Translator(self.model_parameters)
        translator.translator_to_sensor(self.experiment.mesh)

        self.problem.import_sensors_from_metadata(self.model_parameters["MKP_meta_output_path"])
        self.problem.dynamic_solve()        ##TODO: change the name!
        #self.problem.solve()

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
 
#%%    
# =============================================================================
#     def update_input(self, sensor_input):
#         
#         with open(self.dt_path, 'r') as f:
#             dt_params = json.load(f)
#         
#         # currently, only updates rho value
#         if isinstance(sensor_input, (int, float)):
#             dt_params[0]["parameters"]["rho"] = sensor_input
#             
#             with open(self.dt_path, 'w') as file:
#                 json.dump(dt_params, file, indent=4)
#             return True
#         else:
#             return False
# =============================================================================
#%%

    def update_parameters(self, updates, target_name=None):
        """
        Updates the specified parameters in the JSON file.
        """
        try:
            with open(self.dt_path, 'r') as f:
                dt_params = json.load(f)
    
            updated = False
            model_type_params = None
            
            # Update parameters in matching entries
            for entry in dt_params:
                if entry["name"] == target_name:
                    for key, value in updates.items():
                        if key in entry["parameters"]:
                            if entry["parameters"][key] != value:
                                entry["parameters"][key] = value
                                model_type_params = entry
                                updated = True
   
            # Save the updated JSON back to the file
            if updated:
                with open(self.dt_path, 'w') as file:
                    json.dump(dt_params, file, indent=4)
                return True, model_type_params
            else:
                return False, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return False


        
    def solve(self):

        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()
        
        self.sensor_out = self.api_dataFrame['E_plus_080DU_HSN-u-_Avg1'].iloc[-1] # *1000 #Convertion from meter to milimeter
        #self.sensor_out = self.api_dataFrame['E_plus_413TU_HSS-m-_Avg1'].iloc[-1]
 
        vs_file_path = self.model_parameters["virtual_sensor_added_output_path"]
        with open(vs_file_path, 'r') as file:
            self.vs_data = json.load(file)        
        
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
    
    def fields_assignment(self, data):
        if data == None:
            pass
        
        else:
            for i in data.keys():
                if i == "displacement":
                    self.problem.fields.displacement = data[i]
                elif i == "temperature":
                    self.problem.fields.temperature = data[i]
        
    def fields_data_storer(self, path):
        data_to_store = {}

        for i in dir(self.problem.fields):
            if not i.startswith("_"):
                k = getattr(self.problem.fields, i)
                if k:
                    data_to_store[i] = k.x.array[:]
        try:
            pkl_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" + path
            with open(f"{pkl_path}_params.pkl", "wb") as f:
                pickle.dump(data_to_store, f)
                
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

#%%
    
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
