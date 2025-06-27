import random
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

class DigitalTwinInterface:
    def __init__(self, path):
        #self.parameters_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json'
        self.parameters_path = path
        self.available_models = ["TransientThermal_1", "Displacement_1", "Displacement_2"]
        self.current_model = self.available_models[0]
        self.orchestrator = Orchestrator(self.parameters_path, self.current_model)

    def change_model(self, model_name):
        if model_name in self.available_models:
            #self.orchestrator = Orchestrator(self.parameters_path, model_name)
            self.current_model = model_name
            print(f"\nModel changed to: {self.current_model}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def generate_random_input(self, parameter: str) -> dict:
        if parameter == "rho":
            value = random.randint(90 // 5, 160 // 5) * 100
        elif parameter == "E":
            value = random.randint(100 // 5, 225 // 5) * 10**10
        else:
            raise KeyError("Invalid parameter. Choose 'rho' or 'E'.")
        return {parameter: value}

    def run(self, input_dict: dict):
        return self.orchestrator.run(input_dict, self.current_model)

    def run_model(self, action):
        while True:
            print("\nCurrent model:", self.current_model)
            print()
            action = input(f"\nType 'run' to generate and run: {self.current_model},'change' to switch model, or 'exit' to quit: ").strip().lower()

            print(f"\nyou typed {action}")
            
            if action == "run":
                param = input("\nEnter parameter to randomize ('rho' or 'E'): ").strip()
                try:
                    input_values = self.generate_random_input(param)
                    print(f"\nGenerated input: {input_values}")
                    result = self.run(input_values)
                    print("\nRun complete!")
                    #print("Result:", result)
                except Exception as e:
                    print("\nError:", str(e))

            elif action == "change":
                print(f"\nAvailable models: {', '.join(self.available_models)}\n")
                new_model = input(f"\nEnter new model name -> {', '.join(self.available_models)}: ").strip()
                try:
                    self.change_model(new_model)
                except Exception as e:
                    print("\nError:", str(e))

            elif action == "exit":
                print("\nExiting interface.")
                break

            else:
                print("\nUnknown command. Please type 'run', 'change', or 'exit'.")


#dt = DigitalTwinInterface()
#dt.run_model()