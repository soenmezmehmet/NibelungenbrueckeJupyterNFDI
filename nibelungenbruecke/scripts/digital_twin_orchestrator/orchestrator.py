import json
import numpy as np
from nibelungenbruecke.scripts.digital_twin_orchestrator.digital_twin import DigitalTwin

class Orchestrator:
    """
   Manages the workflow of the digital twin, transitioning from a linear, step-by-step approach 
   to a more dynamic, feedback-based system.
   
   This class initializes and orchestrates a digital twin model, enabling predictions and comparisons 
   based on input values.
   
   Attributes:
       model_parameters_path (str): Path to the model parameters dictionary.
       model_to_run (str): Specifies which predefined model to execute.
       updated (bool): Indicates whether the model has been updated based on comparisons.
       digital_twin_model (DigitalTwin): The initialized digital twin model instance.
   
    """
    
    def __init__(self, model_parameters_path: str, model_to_run: str="Displacement_1"):
        """
        Initializes the Orchestrator.
        
        Args:
            model_parameters_path (str): Path to the model parameters dictionary.
            model_to_run (str): Specifies which predefined model to execute. Defaults to "Displacement_1".
        
        """
        self.updated = False
        self.model_to_run = model_to_run
        self.model_parameters_path = model_parameters_path
        
        self.digital_twin_model = self._digital_twin_initializer()
        
    def _digital_twin_initializer(self):
        """
       Initializes the digital twin model.
       
       Returns:
           DigitalTwin: An instance of the DigitalTwin class initialized with the given parameters.
       
        """
        return DigitalTwin(self.model_parameters_path, self.model_to_run)
        
    def predict_dt(self, digital_twin, input_value, model_to_run):   
        """
        Runs "prediction" method of specified digital twin object.
        
        Args:
            digital_twin (DigitalTwin): The digital twin model instance.
            input_value : The input data for prediction.
            model_to_run (str): Specifies which predefined model to execute.
        
        """
        return digital_twin.predict(input_value, model_to_run)
    
    def predict_last_week(self, digital_twin, inputs):
        """
        Generates predictions for a series of inputs from the series of inputs of same data.
        
        Args:
            digital_twin (DigitalTwin): The digital twin model instance.
            inputs (list|dict): A list of input values for prediction.
        
        Returns:
            list: A list of predictions.
        """
        predictions = []
        for input_value in inputs:
            prediction = digital_twin.predict(input_value)
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    def compare(self, output, input_value):
        self.updated = (output == 2 * input_value)


    def run(self, input_value, model_to_run):
        """
        Runs the digital twin model prediction.
        
        TODO:
        - Implement conditional execution based on prediction type.
        - Support more flexible input types.
        
        Args:
            input_value : The input data for prediction.
            model_to_run (str): Specifies which predefined model to execute.
        
        """
                
        prediction = self.predict_dt(self.digital_twin_model, input_value, model_to_run)
        #prediction = self.predict_last_week(digital_twin, input_value)     ##TODO: More flexible input type!!
        print("Prediction:", prediction)

import random

def generate_random_rho(params: dict={}, parameters: str="rho"):
    """
    Generates 'rho' and 'E' values.
    """
    if parameters == "rho":
        #random_value = random.randint(110 // 5, 120 // 5) * 100
        random_value = random.randint(90 // 5, 160 // 5) * 100
    elif parameters == "E":
        random_value = random.randint(100 // 5, 225 // 5) * 10**10
    else:
        raise KeyError("unexpected parameters!")
    
    params[parameters] = random_value
    return params

#%%

if __name__ == "__main__":
    
    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"   
    model_to_run = "TransientThermal_1"
    orchestrator = Orchestrator(path, model_to_run)
   
#####  
  
    input_value=generate_random_rho()
    print(input_value)

    orchestrator.run(input_value, model_to_run)
