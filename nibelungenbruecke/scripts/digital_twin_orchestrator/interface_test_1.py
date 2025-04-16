import random
import os
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

PARAMETERS_PATH = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
AVAILABLE_MODELS = ["TransientThermal_1", "Displacement_1", "Displacement_2"]

# Initializing the orchestrator
orchestrator = Orchestrator(PARAMETERS_PATH, "TransientThermal_1")
current_model = "TransientThermal_1"



def generate_random_input(params: dict = {}, parameter: str = "rho") -> dict:
    """
    Generate random input values for given parameter.
    """
    if parameter == "rho":
        value = random.randint(90 // 5, 160 // 5) * 100
    elif parameter == "E":
        value = random.randint(100 // 5, 225 // 5) * 10**10
    else:
        raise KeyError("Invalid parameter. Choose 'rho' or 'E'.")
    
    params[parameter] = value
    return params

def run_model():
    global orchestrator, current_model
    
    while True:
        print("\n Current model:", current_model)
        action = input("Type 'run' to generate and run input, 'change' to switch model, or 'exit' to quit: ").strip().lower()


        if action == "run":
            param = input("Enter parameter to randomize ('rho' or 'E'): ").strip()
            try:
                input_values = generate_random_input(parameter=param)
                print(f"Generated input: {input_values}")
                orchestrator.run(input_values, current_model)
                print("Run complete.")
            except KeyError as e:
                print(e)

        elif action == "change":
            print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            new_model = input("Enter new model name: ").strip()
            if new_model in AVAILABLE_MODELS:
                orchestrator = Orchestrator(PARAMETERS_PATH, new_model)
                current_model = new_model
                print(f"Model changed to: {current_model}")
            else:
                print("Invalid model name.")

        elif action == "exit":
            print("Exiting interface.")
            break

        else:
            print("Unknown command. Please type 'run', 'change', or 'exit'.")

# Call this function to start the interface
run_model()
