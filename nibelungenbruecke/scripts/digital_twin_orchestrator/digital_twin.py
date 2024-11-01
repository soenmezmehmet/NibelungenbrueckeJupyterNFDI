import sys
from nibelungenbruecke.scripts.digital_twin_orchestrator.displacement_model import DisplacementModel
import json
import importlib
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator_cache import ObjectCache
import pickle
import numpy as np
from nibelungenbruecke.scripts.digital_twin_orchestrator.model_update import UpdateModelState

class DigitalTwin:
    def __init__(self, model_path: str, model_parameters: dict, dt_path: str, model_to_run = "Displacement_1"):
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.dt_path = dt_path
        self.model_to_run = model_to_run
        self.load_models()
        self.cache_object = ObjectCache()
        
    def load_models(self):
        with open(self.dt_path, 'r') as json_file:
            self.models = json.load(json_file)
        
    def set_model(self):
        for model_info in self.models:
            if model_info["name"] == self.model_to_run:
                self.cache_model_name = model_info["type"]
                self.cache_object_name = model_info["class"]
                self.cache_model_path = model_info["path"]
                return True
        return False    #TODO: Currently does not create new models if the model_to_run parametes is not in the dt_path json file!!
            
    def predict(self, input_value):      
        if self.set_model():
            
            if not self.cache_object.cache_model:
                digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                
                if not digital_twin_model:
                    module = importlib.import_module(self.cache_model_name)
                    digital_twin_model = getattr(module, self.cache_object_name)(self.model_path, self.model_parameters, self.dt_path)
                    #sys.path.append(module.__path__)
                    #digital_twin_model.solve()
                    with open(self.cache_model_path, 'wb') as f:
                        pickle.dump(digital_twin_model, f)       
                        
                        
                self.cache_object.cache_model =  digital_twin_model     ##TODO:
                
            else:
                if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
                    digital_twin_model = self.cache_object.cache_model
                    #digital_twin_model = self.restore_model(digital_twin_model, self.model_to_run)
                                       
                else:
                    digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                    #digital_twin_model = self.restore_model(digital_twin_model, self.model_to_run)
                    self.cache_object.cache_model =  digital_twin_model                     
            
            self.cache_object.update_store(digital_twin_model)  ## TODO: Instead of Digital_twin_model, self.cache_object.cache_model could be used?!
            
#%%
            ums = UpdateModelState(digital_twin_model, self.model_to_run, self.model_path)
            digital_twin_model = ums.reconstruct_model()
            

            ums.store_model_state()
                     
#%%
                    
            #if digital_twin_model.update_input(input_value):
                #digital_twin_model.solve()
                #return digital_twin_model.export_output(self.model_to_run)
                
            #return digital_twin_model.export_output(self.model_to_run)
            
        return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters

    #%%
# =============================================================================
#     def store_model_state(self, dm, model_name):
#         displacement_function = dm.problem.fields.displacement
#         displacement_values = displacement_function.x.array[:]
#         mesh_coordinates = dm.problem.mesh.geometry.x[:] 
#         
#         data_to_store = {
#              "displacement_values": displacement_values,
#              "mesh_coordinates": mesh_coordinates,
#              "mesh_topology": dm.problem.mesh.topology.cell_type,
#          
#          }
#         
#         with open(f"{model_name}.pkl", "wb") as f:
#             pickle.dump(data_to_store, f)
#             
#     def load_correct_model(self, model_name):
#         with open(f"{model_name}.pkl", "rb") as f:
#             stored_data = pickle.load(f)  
#             
#         return stored_data
#     
#     def restore_model(self, dm, model_name):
#         stored_data = self.load_correct_model(model_name)
#         displacement_values = stored_data["displacement_values"]
#         mesh_coordinates = stored_data["mesh_coordinates"]
# 
#         dm.experiment.mesh.geometry.x[:, :] = mesh_coordinates
# 
#         degree = 2 
#         dim = 3  
#         V = df.fem.VectorFunctionSpace(dm.experiment.mesh, ("Lagrange", degree))
# 
#         displacement_function = df.fem.Function(V)
#         displacement_function.x.array[:] = displacement_values
# 
#         dm.problem.V = V
#         dm.problem.fields = SolutionFields(displacement=displacement_function)
#         return dm
# =============================================================================
        
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
    
    model_to_run = "Displacement_1"
    dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
    

    input_value=round(2.0*10**11, 1)
    
    dt.predict(input_value)



          
#%%

# =============================================================================
# def predict(self, input_value):      
#     if self.set_model():
#         
#         if not self.cache_object.cache_model:
#             digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
#             
#             if not digital_twin_model:
#                 module = importlib.import_module(self.cache_model_name)
#                 digital_twin_model = getattr(module, self.cache_object_name)(self.model_path, self.model_parameters, self.dt_path)
#                 #sys.path.append(module.__path__)
#                 with open(self.cache_model_path, 'wb') as f:
#                     pickle.dump(digital_twin_model, f)
#                     
#             self.cache_object.cache_model =  digital_twin_model
#             
#         else:
#             if self.cache_object.model_name == self.cache_object_name: #TODO: why??? -> changed now!!
#                 digital_twin_model = self.cache_object.cache_model
#                 
#             else:
#                 digital_twin_model = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
#                 self.cache_object.cache_model =  digital_twin_model                     
#         
#         self.cache_object.update_store(digital_twin_model)
#                 
#         if digital_twin_model.update_input(input_value):
#             digital_twin_model.solve()
#             return digital_twin_model.export_output(self.model_to_run)
#         
#     return None     #TODO: One possible consequence of the above TODO part. Could be enhanced by instead of returning None, create a new object/model with some default parameters
# 
#             
# =============================================================================

    
