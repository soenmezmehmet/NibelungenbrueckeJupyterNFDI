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
    """Managing digital twin models."""
    
    def __init__(self, model_parameters_path: dict, model_to_run = "Displacement_1"): 
        self.model_to_run = model_to_run
        self.orchestrator_parameters = self._extract_model_parameters(model_parameters_path)
        self._load_models()
        self.cache_object = ObjectCache()
        self.initial_model = self._initialize_default_model()
    
    def _extract_model_parameters(self, path):
        """Load parameters from JSON"""
        
        with open(path, 'r') as file:
            _extracted_model_parameters = json.load(file)
        return _extracted_model_parameters
        
    def _initialize_default_model(self):
        """Initialize the digital twin model"""
        
        digital_twin_model = None
        for i in self._models:
            if i["name"] == self.model_to_run:
                model_path = self.orchestrator_parameters["model_path"]
                model_parameters = self.orchestrator_parameters["generation_models_list"][0]["model_parameters"]
                dt_params_path = self.orchestrator_parameters["generation_models_list"][0]["digital_twin_parameters_path"]
                
                module = importlib.import_module(i["type"])
                digital_twin_model = getattr(module, i["class"])(model_path, model_parameters, dt_params_path)
                return digital_twin_model       ##TODO: DT could be return None!!?
                
        if not digital_twin_model:
            raise ValueError(f"Invalid model {digital_twin_model}. digital twin model should not be empty!")
        
    def _load_models(self):
        dt_params_path = self.orchestrator_parameters["generation_models_list"][0]["digital_twin_parameters_path"]
        with open(dt_params_path, 'r') as json_file:
            self._models = json.load(json_file)
    
    def _set_model(self):
        for model_info in self._models:
            if model_info["name"] == self.model_to_run:
                self.cache_model_name = model_info["type"]
                self.cache_object_name = model_info["class"]
                rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/"
                self.cache_model_path = rel_path + model_info["path"]
                return True
        raise ValueError(f"'{self.model_to_run}' not found in the defined models.")
            
    def uploader(self):
        try:
            rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/"
            with open(f"{rel_path}{self.model_to_run}_params.pkl", "rb") as f:
                self.model_params = pickle.load(f)                
                return self.model_params
        
        except FileNotFoundError:
            print(f"Error: The file {self.model_to_run} was not found!")    #TODO: Use assertion instead!!
            return None
        
        except Exception as e:
            print(f"An unexpected error!: {e}")
            
    def predict(self, input_value):      
        if self._set_model():
            
            if not self.cache_object.cache_model:
                parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                
                if not parameters:
                    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
                    with open(path, 'r') as file:
                        parameters = json.load(file)
                                            
                        self.cache_object.cache_model = parameters
                
            else:
                if self.cache_object.model_name == self.cache_object_name:
                    parameters = self.cache_object.cache_model
                    
                else:
                    parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                    self.cache_object.cache_model =  parameters                     
                    
            
            updated, updated_params = self.initial_model.update_parameters(input_value, self.model_to_run)
            if updated:
                
                #self.update_input(input_value)
                self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]["material_parameters"]  = updated_params["parameters"]
                
                self.cache_object.update_store(parameters)
                                               
                self.initial_model.model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
            
                self.initial_model.GenerateModel() ##TODO: Avoid loading the mesh every time!!
            
                self.initial_model.field_data_storer(self.model_to_run) ##TODO: Make it conditional. Run it if there is a change!!
            
                self.uploader()
                
                self.initial_model.field_assignment(self.model_params)
                
                self.initial_model.solve()
                self.initial_model.field_data_storer(self.model_to_run)

            return self.initial_model.export_output(self.model_to_run)
            
        return None
    
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

import random

def generate_random_rho():
    """
    Generates a random 'rho' value for vehicles passing through the bridge 
    between 5000 and 10000,
    """
    params = dict()
    random_value = random.randint(5000 // 50, 10000 // 50) * 50
    params["rho"] =random_value
    
    return params



if __name__ == "__main__":
    
    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
    
    model_to_run = "Displacement_1"
    #dt = DigitalTwin(model_path, model_parameters, dt_path, model_to_run)
    dt = DigitalTwin(path, model_to_run)  
    
    #input_value=round(2.0*10**11, 1)
    input_value = generate_random_rho()
    
    dt.predict(input_value)

