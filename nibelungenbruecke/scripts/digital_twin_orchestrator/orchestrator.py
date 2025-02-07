import json
import numpy as np
from nibelungenbruecke.scripts.digital_twin_orchestrator.digital_twin import DigitalTwin

class Orchestrator:
    def __init__(self, model_parameters_path: str, model_to_run: str="Displacement_1"):
        self.updated = False
        self.model_to_run = model_to_run
        self.model_parameters_path = model_parameters_path
        
        self.digital_twin_model = self._digital_twin_initializer()
        
    def _digital_twin_initializer(self):
        return DigitalTwin(self.model_parameters_path, self.model_to_run)
        
    def predict_dt(self, digital_twin, input_value, model_to_run):
        return digital_twin.predict(input_value, model_to_run)
    
    def predict_last_week(self, digital_twin, inputs):
        predictions = []
        for input_value in inputs:
            prediction = digital_twin.predict(input_value)
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    def compare(self, output, input_value): #TODO: !!
        self.updated = (output == 2 * input_value)


    def run(self, input_value, model_to_run):   ##TODO: Conditional run based on prediction type!!
                
        prediction = self.predict_dt(self.digital_twin_model, input_value, model_to_run)
        #prediction = self.predict_last_week(digital_twin, input_value)     ##TODO: More flexible input type!!
        print("Prediction:", prediction) #TODO

#%%
if __name__ == "__main__":
    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"   
    model_to_run = "Displacement_2"
    orchestrator = Orchestrator(path, model_to_run)

    #input_value=[round(2.0*10**11, 1), round(2.7*10**11, 1), round(3.4*10**11, 1), round(4.0*10**11, 1)]
    #input_value=round(2.0*10**11, 1)
    
    import random

    def generate_random_rho(params: dict={}, parameters: str="rho"):
        """
        Generates a random 'rho' value for vehicles passing through the bridge 
        between 5000 and 10000,
        """
        if parameters == "rho":
            #random_value = random.randint(110 // 5, 120 // 5) * 100
            random_value = random.randint(90 // 5, 160 // 5) * 100
        elif parameters == "E":
            random_value = random.randint(100 // 5, 225 // 5) * 10**10
        else:
            raise KeyError("unexpected parameters! Please use a check the parameters!")
        
        params[parameters] = random_value
        return params
    
#####  
  
    input_value=generate_random_rho()
    print(input_value)

    #orchestrator.run(input_value, model_to_run)
      
# #####

    model_to_run = "Displacement_1"
    input_value=generate_random_rho(input_value, parameters="E")
    print(input_value)
    
    orchestrator.run(input_value, model_to_run) 
    
# #####

#     model_to_run = "Displacement_3"
#     input_value_04=generate_random_rho()
#     print(input_value_04)

#     orchestrator.run(input_value, model_to_run)

"""
def run(self, input_value, model_to_run):   ##TODO: Make it conditional for two prediction kinds!!

    
    
    if model_to_run != self.model_to_run:   ##TODO: To be done in DT!!
        self.model_to_run = model_to_run  
        if self.model_to_run in self.digital_twin_model.digital_twin_models.keys():
            self.DT_model = self.digital_twin_model
            
        self.digital_twin_model = self._digital_twin_initializer()
        
    prediction = self.predict_dt(self.digital_twin_model, input_value)
    #prediction = self.predict_last_week(digital_twin, input_value)     ##TODO: input type can be better!!
    print("Prediction:", prediction) #TODO
"""