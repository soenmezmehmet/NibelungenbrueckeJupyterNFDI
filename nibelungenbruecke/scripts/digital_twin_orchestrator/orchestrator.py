from nibelungenbruecke.scripts.digital_twin_orchestrator.digital_twin import DigitalTwin
import json

class Orchestrator:
    def __init__(self, model_parameters_path: str, model_to_run="Displacement_1"):
        self.updated = False
        self.model_parameters_path = model_parameters_path
        self.model_to_run = model_to_run
        
        self.digital_twin_models = self._digital_twin_initializer()
        
    def _digital_twin_initializer(self):
        return DigitalTwin(self.model_parameters_path, self.model_to_run)
        
        
    def predict_dt(self, digital_twin, input_value):
        return digital_twin.predict(input_value)
    
    def predict_last_week(self, digital_twin, inputs):
        predictions = []
        for input_value in inputs:
            prediction = digital_twin.predict(input_value)
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    def compare(self, output, input_value): #TODO
        self.updated = (output == 2 * input_value)

    def run(self, input_value, model_to_run):
        prediction = self.predict_dt(self.digital_twin_models, input_value)
        #prediction = self.predict_last_week(digital_twin, input_value)
        print("Prediction:", prediction) #TODO
        
#%%
if __name__ == "__main__":
    path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"   
    model_to_run = "Displacement_2"
    orchestrator = Orchestrator(path, model_to_run)

    #input_value=[round(2.0*10**11, 1), round(2.7*10**11, 1), round(3.4*10**11, 1), round(4.0*10**11, 1)]
    #input_value=round(2.0*10**11, 1)
    
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
    
    
    input_value=generate_random_rho()
    print(input_value)

    orchestrator.run(input_value, model_to_run)
#####
    input_value_02=generate_random_rho()
    print(input_value_02)
    
    orchestrator.run(input_value, model_to_run)
    
#####

    model_to_run = "Displacement_1"
    input_value_03=generate_random_rho()
    print(input_value_03)

    orchestrator.run(input_value, model_to_run)
    
#####

    model_to_run = "Displacement_3"
    input_value_04=generate_random_rho()
    print(input_value_04)

    orchestrator.run(input_value, model_to_run)