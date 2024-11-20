import sys
import ufl
import json
import pickle
import importlib
import numpy as np
import dolfinx as df
from mpi4py import MPI

from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator_cache import ObjectCache
from nibelungenbruecke.scripts.digital_twin_orchestrator.displacement_model import DisplacementModel

class DigitalTwin:
    def __init__(self, model_parameters_path: str, model_to_run = "Displacement_1"):    
        with open(model_parameters_path, 'r') as file:
            self.orchestrator_parameters = json.load(file)
            
        self.model_path = self.orchestrator_parameters["model_path"]
        self.model_parameters = self.orchestrator_parameters["generation_models_list"][0]["model_parameters"]
        self.dt_path = self.orchestrator_parameters["generation_models_list"][0]["digital_twin_parameters_path"]
        
        self.displacement_model = DisplacementModel(self.model_path, self.model_parameters, self.dt_path)

        self.model_to_run = model_to_run
        #self.load_models()
        self.cache_object = ObjectCache()
        
#%%        
# =============================================================================
#     def load_models(self):
#         with open(self.dt_path, 'r') as json_file:
#             self.models = json.load(json_file)
# =============================================================================
        
#%%
    def set_model(self):
        for model_info in self.models:
            if model_info["name"] == self.model_to_run:
                self.cache_model_name = model_info["type"]
                self.cache_object_name = model_info["class"]
                rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/"
                self.cache_model_path = rel_path + model_info["path"]
                return True
        raise ValueError(f"'{self.model_to_run}' not found in the defined models.")
 
 #%%           
    def uploader(self): #TODO: Make it with for loop!!
        try:
            rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/"
            with open(f"{rel_path}{self.model_to_run}_params.pkl", "rb") as f:
            #with open(f"{self.model_to_run}_params.pkl", "rb") as f:
                self.model_params = pickle.load(f)                
                return self.model_params
        
        except FileNotFoundError:
            print(f"Error: The file {self.model_to_run} was not found!")    #TODO: Use assertion instead!!
            return None
        
        except Exception as e:
            print(f"An unexpected error!: {e}")
            
#%%            
# =============================================================================
#     def updater(self, dm):
#         if self.uploader():
#             self.extracter(dm)
#             
#         else:
#             dm.experiment.mesh = self.mesh
#             dm.solve()
# =============================================================================
#%%    
# =============================================================================
#     def extracter(self, dm):
#         displacement_values = self.model_params["displacement"]
#         temperature_values = self.model_params["temperature"]
#         #mesh_coordinates = self.model_params["mesh_coordinates"]
#         
#         degree = 2
#         dim = 3
#         V = df.fem.VectorFunctionSpace(dm.problem.mesh, ("Lagrange", degree))
#         
#         displacement_function = df.fem.Function(V)
#         displacement_function.x.array[:] = displacement_values
#         
#         #dm.problem.mesh = self.mesh
#         dm.problem.V = V
#         #dm.problem.fields = SolutionFields(displacement=displacement_function)
#         #dm.solve()
#         return dm
# =============================================================================
    
#%%
    
    def storer(self, dm):
        
        data_to_store = {}

        for i in dir(dm.problem.fields):
            if not i.startswith("_"):
                k = getattr(dm.problem.fields, i)
                if k:
                    data_to_store[i] = k.x.array[:]
        try:
            pkl_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" + self.model_to_run
            with open(f"{pkl_path}_params.pkl", "wb") as f:
                pickle.dump(data_to_store, f)
                
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            
    def update_input(self, sensor_input):
        
        #TODO: Make this part more automated/flexible!  
        if isinstance(sensor_input, (int, float)):
            ##TODO: Change to rho!!
            self.cache_object.cache_model['generation_models_list'][0]['model_parameters']['material_parameters']["E"] = sensor_input
            self.cache_object.cache_model['generation_models_list'][1]['model_parameters']["material_parameters"]["E"] = sensor_input
        else:
            pass
    
    #%%        
    def predict(self, input_value):      
        if self.set_model():
            
            if not self.cache_object.cache_model:
                parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                
                if not parameters:
                    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
                    with open(path, 'r') as file:
                        parameters = json.load(file)
                                            
                self.cache_object.cache_model = parameters
                
            else:
                if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
                    parameters = self.cache_object.cache_model
                    
                else:
                    parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                    self.cache_object.cache_model =  parameters                     
                    
            self.update_input(input_value)
                        
            self.cache_object.update_store(parameters)

#%%            
# =============================================================================
#             self.model_path = self.cache_object.cache_model["model_path"]
#             self.model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
#             self.dt_path = self.cache_object.cache_model["generation_models_list"][0]["digital_twin_parameters_path"]
#             
#             digital_twin_model = DisplacementModel(self.model_path, self.model_parameters, self.dt_path)
#             digital_twin_model.GenerateModel()
# =============================================================================
#%%            
            self.displacement_model.model_path = self.cache_object.cache_model["model_path"]
            self.displacement_model.model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
            self.displacement_model.dt_path = self.cache_object.cache_model["generation_models_list"][0]["digital_twin_parameters_path"]
            
            self.displacement_model.GenerateModel()
            
            digital_twin_model = self.displacement_model
            
            self.storer(digital_twin_model)
            
            ##LinearElasticityJob
            self.uploader()
            #self.extracter(digital_twin_model)
            digital_twin_model.field_assignment(self.model_params)

            digital_twin_model.solve()
            self.storer(digital_twin_model)

            return digital_twin_model.export_output(self.model_to_run)
            
        return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters
    
    def store_update(self):            
        measured_vs_path = self.model_parameters["virtual_sensor_added_output_path"]
        with open(measured_vs_path, 'r') as f:
            sensor_measurement = json.load(f)
            
        triggered = False    
        for i in sensor_measurement["virtual_sensors"].keys():
            if sensor_measurement["virtual_sensors"][i]["displacements"][-1] == sensor_measurement["virtual_sensors"][i]["displacements"][-2]:
                triggered = False
            else:
                triggered = True
                
        return triggered
     
#%%

if __name__ == "__main__":
    
    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
    
    model_to_run = "Displacement_1"
    #dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
    dt = DigitalTwin(path, model_to_run)    

    input_value=round(2.0*10**11, 1)
    
    dt.predict(input_value)


    
#%%    
# =============================================================================
#     
#     model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
#     
#     model_parameters = {'model_name': 'displacements',
#      'df_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_df_output.csv',
#      'meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_meta_output.json',
#      'MKP_meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_meta_output.json',
#      'MKP_translated_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_translated.json',
#      'virtual_sensor_added_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/virtual_sensor_added_translated.json',
#      'cache_path': '',
#      'paraview_output': True,
#      'paraview_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview',
#      'material_parameters': {'E': 40000000000000.0, 'nu': 0.2, 'rho': 2350},
#      'tension_z': 0.0,
#      'mass': 50000.0,
#      'g': 9.81,
#      'initial_position': [0.0, 0.0, 0.0],
#      'speed': 1.0,
#      'length': 7.5,
#      'width': 2.5,
#      'height': 6.5,
#      'length_road': 95.0,
#      'width_road': 14.0,
#      'thickness_deck': 0.2,
#      'dt': 30.0,
#      'reference_temperature': 300,
#      'temperature_coefficient': 1e-05,
#      'temperature_alpha': 1e-05,
#      'temperature_difference': 5.0,
#      'reference_height': -2.5,
#      'boundary_conditions': {'bc1': {'model': 'clamped_edge',
#        'side_coord_1': 0.0,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1},
#       'bc2': {'model': 'clamped_edge',
#        'side_coord_1': 95.185,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1}}}
#     
#     dt_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_parameters.json'
#     
#     model_to_run = "Displacement_1"
#     #dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
#     dt = DigitalTwin(path, model_to_run)    
# 
#     input_value=round(2.0*10**11, 1)
#     
#     dt.predict(input_value)
# 
# =============================================================================

#%%

# =============================================================================
# import sys
# import ufl
# import json
# import pickle
# import importlib
# import numpy as np
# import dolfinx as df
# from mpi4py import MPI
# 
# #from nibelungenbruecke.scripts.digital_twin_orchestrator.model_update import UpdateModelState
# from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator_cache import ObjectCache
# from nibelungenbruecke.scripts.digital_twin_orchestrator.displacement_model import DisplacementModel
# from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
# 
# class DigitalTwin:
#     def __init__(self, model_parameters_path: str, model_to_run = "Displacement_1"):    
#    #def __init__(self, orchestrator_parameters: dict, model_path: str, model_parameters: dict, dt_path: str, model_to_run = "Displacement_1"):
#         #self.orchestrator_parameters = orchestrator_parameters
#         with open(model_parameters_path, 'r') as file:
#             self.orchestrator_parameters = json.load(file)
#             
#         #self.model_path = self.orchestrator_parameters["model_path"]
#         #self.model_parameters = self.orchestrator_parameters["generation_models_list"][0]["model_parameters"]
#         #self.dt_path = self.orchestrator_parameters["generation_models_list"][0]["digital_twin_parameters_path"]
# 
#         self.model_path = model_path
#         self.model_parameters = model_parameters
#         self.dt_path = dt_path
#         self.model_to_run = model_to_run
#         self.load_models()
#         self.cache_object = ObjectCache()
#         
#     def load_models(self):
#         with open(self.dt_path, 'r') as json_file:
#             self.models = json.load(json_file)
#         
#     def set_model(self):
#         for model_info in self.models:
#             if model_info["name"] == self.model_to_run:
#                 self.cache_model_name = model_info["type"]
#                 self.cache_object_name = model_info["class"]
#                 self.cache_model_path = model_info["path"]
#                 return True
#         return False    #TODO: Currently does not create new models if the model_to_run parametes is not in the dt_path json file!!
#             
#     def uploader(self, cache_path, model_name): #TODO: Make it with for loop!!
#         try:
#             with open(f"{self.model_to_run}_params.pkl", "rb") as f:
#                 self.model_params = pickle.load(f)
#                 self.model_params["displacement_values"] = np.array(self.model_params["displacement_values"])
#                 self.model_params["mesh_coordinates"] = np.array(self.model_params["mesh_coordinates"])
#                 
#                 return self.model_params
#         
#         except FileNotFoundError:
#             print(f"Error: The file {self.model_to_run} was not found!")    #TODO: Use assertion instead!!
#             return None
#         
#         except Exception as e:
#             print(f"An unexpected error!: {e}")
#             
#             
#     def updater(self, dm):
#         if self.uploader():
#             self.extracter(dm)
#             
#         else:
#             dm.experiment.mesh = self.mesh
#             dm.solve()
#     
#     def extracter(self, dm):
#         displacement_values = self.model_params["displacement_values"]
#         mesh_coordinates = self.model_params["mesh_coordinates"]
#         
#         degree = 2
#         dim = 3
#         V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", degree))
#         
#         displacement_function = df.fem.Function(V)
#         displacement_function.x.array[:] = displacement_values
#         
#         dm.problem.mesh = self.mesh
#         dm.problem.V = V
#         dm.problem.fields = SolutionFields(displacement=displacement_function)
#         dm.solve()
#         return dm
#     
#     def storer(self, dm):
#         
#         data_to_store = {}
# 
#         for i in dir(dm.problem.fields):
#             if not i.startswith("_"):
#                 k = getattr(dm.problem.fields, i)
#                 if k:
#                     data_to_store[i] = k.x.array[:]
#         try:
#             with open(f"{self.model_to_run}_params.pkl", "wb") as f:
#                 pickle.dump(data_to_store, f)
#                 
#         except Exception as e:
#             print(f"An error occurred while saving the model: {e}")
#             
#     def predict(self, input_value):      
#         if self.set_model():
#             
#             if not self.cache_object.cache_model:
#                 parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
#                 
#                 if not parameters:
#                     path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
#                     with open(path, 'r') as file:
#                         parameters = json.load(file)
#                                             
#                 self.cache_object.cache_model = parameters
#                 
#             else:
#                 if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
#                     parameters = self.cache_object.cache_model
#                     
#                 else:
#                     parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
#                     self.cache_object.cache_model =  parameters                     
#                     
#             self.cache_object.update_store(parameters)
#             
#             self.model_path = self.cache_object.cache_model["model_path"]
#             self.model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
#             self.dt_path = self.cache_object.cache_model["generation_models_list"][0]["digital_twin_parameters_path"]
#             
#             digital_twin_model = DisplacementModel(self.model_path, self.model_parameters, self.dt_path)
#             #solved_dm = digital_twin_model.solve()
#             digital_twin_model.solve()
#             self.storer(digital_twin_model)
#             
#             
#             
#             
#             
#             
#             
#             
#             
#             
#             
#             #%%
#                
#             if parameters.update_input(input_value):
#                 parameters.solve()
#                 return parameters.export_output(self.model_to_run)
#             
#         return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters
# 
#                     
#                     
#                 
#                 
#                 
#                     
#                     
#                     
#                     
#             
#             
#             
#             #%%
#             
# # =============================================================================
# #             if not self.cache_object.cache_model:
# #                 digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
# #                 
# #                 if not digital_twin_model:
# #                     module = importlib.import_module(self.cache_model_name)
# #                     digital_twin_model = getattr(module, self.cache_object_name)(self.model_path, self.model_parameters, self.dt_path)
# #                     #sys.path.append(module.__path__)
# #                     #with open(self.cache_model_path, 'wb') as f:
# #                         #pickle.dump(digital_twin_model, f)
# #                         
# #                     with open(self.cache_model_path, "w") as f:
# #                         json.dump(self.orchestrator_parameters, f)
# #                         
# #                 self.cache_object.cache_model =  digital_twin_model
# #                 
# #             else:
# #                 if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
# #                     digital_twin_model = self.cache_object.cache_model
# #                     
# #                 else:
# #                     digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
# #                     self.cache_object.cache_model =  digital_twin_model                     
# #             
# #             self.cache_object.update_store(digital_twin_model)
# #                     
# #             if digital_twin_model.update_input(input_value):
# #                 digital_twin_model.solve()
# #                 return digital_twin_model.export_output(self.model_to_run)
# #             
# #         return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters
# # 
# #                 
# #             
# #             
# # =============================================================================
#         
#         
#         
#      
# #%%
# 
# if __name__ == "__main__":
#     
#     path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
#     
#     model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
#     
#     model_parameters = {'model_name': 'displacements',
#      'df_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_df_output.csv',
#      'meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_meta_output.json',
#      'MKP_meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_meta_output.json',
#      'MKP_translated_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_translated.json',
#      'virtual_sensor_added_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/virtual_sensor_added_translated.json',
#      'cache_path': '',
#      'paraview_output': True,
#      'paraview_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview',
#      'material_parameters': {'E': 40000000000000.0, 'nu': 0.2, 'rho': 2350},
#      'tension_z': 0.0,
#      'mass': 50000.0,
#      'g': 9.81,
#      'initial_position': [0.0, 0.0, 0.0],
#      'speed': 1.0,
#      'length': 7.5,
#      'width': 2.5,
#      'height': 6.5,
#      'length_road': 95.0,
#      'width_road': 14.0,
#      'thickness_deck': 0.2,
#      'dt': 30.0,
#      'reference_temperature': 300,
#      'temperature_coefficient': 1e-05,
#      'temperature_alpha': 1e-05,
#      'temperature_difference': 5.0,
#      'reference_height': -2.5,
#      'boundary_conditions': {'bc1': {'model': 'clamped_edge',
#        'side_coord_1': 0.0,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1},
#       'bc2': {'model': 'clamped_edge',
#        'side_coord_1': 95.185,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1}}}
#     
#     dt_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_parameters.json'
#     
#     model_to_run = "Displacement_1"
#     #dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
#     dt = DigitalTwin(path, model_to_run)    
# 
#     input_value=round(2.0*10**11, 1)
#     
#     dt.predict(input_value)
# 
# 
# 
# =============================================================================


#%%

# =============================================================================
# 
# import sys
# import ufl
# import json
# import pickle
# import importlib
# import numpy as np
# import dolfinx as df
# from mpi4py import MPI
# 
# #from nibelungenbruecke.scripts.digital_twin_orchestrator.model_update import UpdateModelState
# from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator_cache import ObjectCache
# from nibelungenbruecke.scripts.digital_twin_orchestrator.displacement_model import DisplacementModel
# from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
# 
# class DigitalTwin:
#     def __init__(self, model_path: str, model_parameters: dict, dt_path: str, model_to_run = "Displacement_1"):
#         self.model_path = model_path
#         self.model_parameters = model_parameters
#         self.dt_path = dt_path
#         self.model_to_run = model_to_run
#         self.load_models()
#         self.cache_object = ObjectCache()
#         
#     def load_models(self):
#         with open(self.dt_path, 'r') as json_file:
#             self.models = json.load(json_file)
#         
#     def set_model(self):
#         for model_info in self.models:
#             if model_info["name"] == self.model_to_run:
#                 self.cache_model_name = model_info["type"]
#                 self.cache_object_name = model_info["class"]
#                 self.cache_model_path = model_info["path"]
#                 return True
#         return False    #TODO: Currently does not create new models if the model_to_run parametes is not in the dt_path json file!!
#             
#     def uploader(self, cache_path, model_name): #TODO: Make it with for loop!!
#         try:
#             with open(f"{self.model_to_run}_params.pkl", "rb") as f:
#                 self.model_params = pickle.load(f)
#                 self.model_params["displacement_values"] = np.array(self.model_params["displacement_values"])
#                 self.model_params["mesh_coordinates"] = np.array(self.model_params["mesh_coordinates"])
#                 
#                 return self.model_params
#         
#         except FileNotFoundError:
#             print(f"Error: The file {self.model_to_run} was not found!")    #TODO: Use assertion instead!!
#             return None
#         
#         except Exception as e:
#             print(f"An unexpected error!: {e}")
#             
#             
#     def updater(self, dm):
#         if self.uploader():
#             self.extracter(dm)
#             
#         else:
#             dm.experiment.mesh = self.mesh
#             dm.solve()
#     
#     def extracter(self, dm):
#         displacement_values = self.model_params["displacement_values"]
#         mesh_coordinates = self.model_params["mesh_coordinates"]
#         
#         degree = 2
#         dim = 3
#         V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", degree))
#         
#         displacement_function = df.fem.Function(V)
#         displacement_function.x.array[:] = displacement_values
#         
#         dm.problem.mesh = self.mesh
#         dm.problem.V = V
#         dm.problem.fields = SolutionFields(displacement=displacement_function)
#         dm.solve()
#         return dm
#     
#     def storer(self, dm):
#         
#         data_to_store = {}
# 
#         for i in dir(dm.problem.fields):
#             if not i.startswith("_"):
#                 k = getattr(dm.problem.fields, i)
#                 if k:
#                     data_to_store[i] = k.x.array[:]
#         try:
#             with open(f"{self.model_to_run}_params.pkl", "wb") as f:
#                 pickle.dump(data_to_store, f)
#                 
#         except Exception as e:
#             print(f"An error occurred while saving the model: {e}")
#             
#     def predict(self, input_value):
#         
#         
#      
# #%%
# 
# if __name__ == "__main__": 
#     
#     model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
#     
#     model_parameters = {'model_name': 'displacements',
#      'df_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_df_output.csv',
#      'meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_meta_output.json',
#      'MKP_meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_meta_output.json',
#      'MKP_translated_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_translated.json',
#      'virtual_sensor_added_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/virtual_sensor_added_translated.json',
#      'cache_path': '',
#      'paraview_output': True,
#      'paraview_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview',
#      'material_parameters': {'E': 40000000000000.0, 'nu': 0.2, 'rho': 2350},
#      'tension_z': 0.0,
#      'mass': 50000.0,
#      'g': 9.81,
#      'initial_position': [0.0, 0.0, 0.0],
#      'speed': 1.0,
#      'length': 7.5,
#      'width': 2.5,
#      'height': 6.5,
#      'length_road': 95.0,
#      'width_road': 14.0,
#      'thickness_deck': 0.2,
#      'dt': 30.0,
#      'reference_temperature': 300,
#      'temperature_coefficient': 1e-05,
#      'temperature_alpha': 1e-05,
#      'temperature_difference': 5.0,
#      'reference_height': -2.5,
#      'boundary_conditions': {'bc1': {'model': 'clamped_edge',
#        'side_coord_1': 0.0,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1},
#       'bc2': {'model': 'clamped_edge',
#        'side_coord_1': 95.185,
#        'coord_1': 2,
#        'side_coord_2': 0.0,
#        'coord_2': 1}}}
#     
#     dt_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_parameters.json'
#     
#     model_to_run = "Displacement_1"
#     dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
#     
# 
#     input_value=round(2.0*10**11, 1)
#     
#     dt.predict(input_value)
# 
# 
# 
#           
# #%%
# =============================================================================
# # 
# # def predict(self, input_value):      
# #     if self.set_model():
# #         
# #         if not self.cache_object.cache_model:
# #             digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
# #             
# #             if not digital_twin_model:
# #                 module = importlib.import_module(self.cache_model_name)
# #                 digital_twin_model = getattr(module, self.cache_object_name)(self.model_path, self.model_parameters, self.dt_path)
# #                 #sys.path.append(module.__path__)
# #                 with open(self.cache_model_path, 'wb') as f:
# #                     pickle.dump(digital_twin_model, f)
# #                     
# #             self.cache_object.cache_model =  digital_twin_model
# #             
# #         else:
# #             if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
# #                 digital_twin_model = self.cache_object.cache_model
# #                 
# #             else:
# #                 digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
# #                 self.cache_object.cache_model =  digital_twin_model                     
# #         
# #         self.cache_object.update_store(digital_twin_model)
# #                 
# #         if digital_twin_model.update_input(input_value):
# #             digital_twin_model.solve()
# #             return digital_twin_model.export_output(self.model_to_run)
# #         
# #     return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters
# # 
# =============================================================================
#             
# 
#     
# 
# 
# =============================================================================
