import os
import sys
import ufl
import json
import copy
import pickle
import importlib
import numpy as np
import dolfinx as df
from mpi4py import MPI
from pathlib import Path

from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator_cache import ObjectCache


class DigitalTwin:
    """
    Manages digital twin models, handling model initialization, predictions, caching, 
    and parameter updates.
    
    Attributes:
        model_to_run (str): Specifies which predefined model to execute.
        orchestrator_parameters (dict): Stores parameters extracted from the JSON model file.
        digital_twin_models (dict): Stores initialized digital twin models.
        cache_object (ObjectCache): Manages caching of models and parameters.
        
    """
    
    def __init__(self, model_parameters_path: dict, model_to_run = "Displacement_1"): 
        """
       Initializes the DigitalTwin instance.
       
       Args:
           model_parameters_path (str): Path to the JSON file containing model parameters.
           model_to_run (str): Specifies which predefined model to execute. Defaults to "Displacement_1".
           
       """
        self.model_to_run = model_to_run
        self.orchestrator_parameters = self._extract_model_parameters(model_parameters_path)
        self._load_models()
        self.cache_object = ObjectCache()
        self.digital_twin_models = {}
    
    def _extract_model_parameters(self, path):
        """
        Loads parameters from a JSON file.
        
        Args:
            path (str): Path to the JSON file.
        
        Returns:
            dict: Extracted model parameters.
        """
        
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {path} was not found!")
        except json.JSONDecodeError:
            raise ValueError("Error: Failed to parse JSON file!")
            
    def _load_models(self):
        """
        Loads the predefined model parameters from the json file.
        
        """
        dt_params_path = self.orchestrator_parameters["generation_models_list"][0]["digital_twin_parameters_path"]      
        try:
            with open(dt_params_path, 'r') as json_file:
                self._models = json.load(json_file)
        except Exception as exc:
            try:
                dt_params_path = dt_params_path.strip("../")
                with open(dt_params_path, 'r') as json_file:
                    self._models = json.load(json_file)
            except:
                raise RuntimeError('Failed to open the path!') from exc
        
    def predict(self, input_value, model_to_run, api_key):
        """
        Predicts the outcome based on the input value by setting up and running a model.
        
        Args:
            input_value (list|dict): Input data for prediction.
            model_to_run (str): The model to execute.
        

        """
        self.model_to_run = model_to_run
        
        if not self._set_model():
            #self.digital_twin_model = self._initialize_default_model()  ##TODO:
            #return 
            raise f"There isn't any predefined model with name {self.model_to_run}. Please check the name or add the model to model parameters"
            
        # Load cached parameters or default parameters if cache is missing
        if self.model_to_run not in self.digital_twin_models.keys():
            self._loaded_params = self._get_or_load_parameters()
            self.initial_model = self._initialize_default_model(api_key)
        else:
            self.initial_model = self.digital_twin_models[self.model_to_run]
            
        # Updates model parameters if necessary
        updated, updated_params = self.initial_model.update_parameters(input_value, self.model_to_run)
        if updated:
            self._update_cached_model(self._loaded_params, updated_params)  # updates model parameters w.r.t. new input data!
            self._run_model(api_key)
        else:
            print()
            return ("Same model with the same parameters!!")
            
        return self.initial_model
    
    def _set_model(self):
        """
        Sets up the model based on predefined configurations.
        
        Returns:
            bool: True if the model is successfully set, otherwise raises an error.
        
        Raises:
            ValueError: If the specified model is not found in the JSON file.
        """

        for model_info in self._models:
            if model_info["name"] == self.model_to_run:
                self.cache_model_name = model_info["type"]
                self.cache_object_name = model_info["class"]
                rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" ##TODO:
                self.cache_model_path = rel_path + model_info["path"]
                return True
        raise ValueError(f"'{self.model_to_run}' not found in the defined models.")     ##TODO: Create a new model !!
    
    def _get_or_load_parameters(self):
        """
        Retrieves cached model parameters or loads default parameters if the cache is missing.
                
        Returns:
            dict: Model parameters.
        """
        if not self.cache_object.cache_model:
            parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
            if not parameters:
                with open(self._default_parameters_path(), 'r') as file:
                    parameters = json.load(file)
                self.cache_object.cache_model = parameters
        else:
            if self.cache_object.model_name != self.cache_object_name:
                parameters = self.cache_object.load_cache(self.cache_model_path, self.cache_model_name)
                if not parameters:
                    with open(self._default_parameters_path(), 'r') as file:
                        parameters = json.load(file)
                        
                self.cache_object.cache_model = parameters
            else:
                parameters = self.cache_object.cache_model
        
        return parameters
    
    def _initialize_default_model(self, api_key):
        """
        Initializes the digital twin model from the default parameters.
        
        Returns:
            object: An instance of the selected digital twin model.
        
        Raises:
            ValueError: If the selected model is not found.
        """
        
        digital_twin_model = None
        for i in self._models:
            if i["name"] == self.model_to_run:      ##TODO: work on the default_parameter JSON file!!
                if "TransientThermal" in self.model_to_run:
                    model_path = self.cache_object.cache_model["model_path"][0]["transientthermal_model_path"]
                elif "Displacement" in self.model_to_run:
                    model_path = self.cache_object.cache_model["model_path"][0]["displacement_model_path"]
                else:
                    raise f"Default parameter file does not have that {self.model_to_run}. Check the JSON file!"
                    
                model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
                dt_params_path = self.cache_object.cache_model["generation_models_list"][0]["digital_twin_parameters_path"]
                """
                module = importlib.import_module(i["type"])
                """
# =============================================================================
#                 try:
#                     # Try importing without changing path
#                     module = importlib.import_module(i["type"])
#                 
#                 except ModuleNotFoundError:
#                     try:
#                         base_dir = Path(__file__).parent.resolve()  # folder where this script lives
#                         if str(base_dir) not in sys.path:
#                             sys.path.insert(0, str(base_dir))
# 
#                         modules = []
#                         for item in os.listdir(base_dir):
#                             item_path = os.path.join(base_dir, item)
#                             # Check if it's a Python file (module) or a package folder
#                             if item.endswith('.py'):
#                                 modules.append(item[:-3])  # strip .py extension
#                             elif os.path.isdir(item_path) and '__init__.py' in os.listdir(item_path):
#                                 modules.append(item)
#                         
#                         
#                         module = importlib.import_module(i["type"])
# 
#                 
#                     except Exception as e:
#                         # Shows full error traceback from the second failure
#                         raise ImportError(
#                             f"Module '{i['type']}' could not be imported even after adjusting the path."
#                         ) from e
# =============================================================================
                #%%                
                module = importlib.import_module(i["type"])
                #%%
                
                digital_twin_model = getattr(module, i["class"])(model_path, model_parameters, dt_params_path)
                digital_twin_model.GenerateModel()
                
                self.digital_twin_models[self.model_to_run] = digital_twin_model
                return digital_twin_model
                
        if not digital_twin_model:
            raise ValueError(f"Invalid model {digital_twin_model}. digital twin model should not be empty!")
        

    
    def _update_cached_model(self, parameters, updated_params):
        """
        Updates the cached model with new parameters and stores the changes.
        
        """
        self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]["material_parameters"] = updated_params["parameters"]
        self.cache_object.update_store(parameters)
        self.initial_model.model_parameters = self.cache_object.cache_model["generation_models_list"][0]["model_parameters"]
         
    def _run_model(self, api_key):
        """
        Extracts latest version of the model that saved last time.
        
        This provides flexibility to switch between different models, allowing for the assignment 
        of field data without having to recreate the models from scratch.
        
        """
        self.uploader()
        self.initial_model.fields_assignment(self.model_params)
        self.initial_model.solve(api_key)
        self.initial_model.fields_data_storer(self.model_to_run)
        
    def uploader(self):
        """
        Loads model parameters from a serialized pickle file.
        
        Returns:
            dict or None: Loaded model parameters or None if the file is missing.
        """
        try:
            ##TODO: 
            rel_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/"
            with open(f"{rel_path}{self.model_to_run}_params.pkl", "rb") as f:
                self.model_params = pickle.load(f)                
                return self.model_params
        
        except FileNotFoundError:
            self.model_params = None
            print(f"Error: The file {self.model_to_run} was not found!")    #TODO: Use assertion instead!!
            return None
        
        except Exception as e:
            print(f"An unexpected error!: {e}")
    
    def _default_parameters_path(self):
        """
        Returns the default parameters file path.
        """
        default_parameters_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, default_parameters_path)
        json_path = os.path.normpath(json_path)
        
        return json_path
   
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
    Generates a random parameters (currently only 'rho') value for vehicles passing through the bridge.
    rho: between 5000 and 10000
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

