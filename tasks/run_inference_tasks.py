import json
import os
import sys

# Get the parent directory of the current script
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(root_path)

from scripts.inference.run_inference_problem import run_inference_problem
# from scripts.inference.plot_inference_results import plot_inference_results

# def task_generate_synthetic_data():
#     data_parameters_path = "input/settings/generate_data_parameters.json"
#     with open(data_parameters_path, 'r') as f:
#         data_parameters = json.load(f)

#     return {'actions': [(generate_data,[],{'parameters':data_parameters})],
#             'file_dep': [data_parameters["model_path"]],
#             'targets': [data_parameters["output_path"]+data_parameters["output_format"]],
#             'uptodate': [True]}

if __name__ == "__main__":

    data_parameters_path = "input/settings/inference_parameters.json"
    with open(data_parameters_path, 'r') as f:
        data_parameters = json.load(f)
        
    run_inference_problem(data_parameters)