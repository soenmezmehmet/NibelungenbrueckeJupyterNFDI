import json
import importlib
from scripts.utilities import checks

### CONFIG TASKS ###
doit_parameters_path = "input/settings/doit_parameters.json"
checks.assert_path_exists(doit_parameters_path)

with open(doit_parameters_path, 'r') as file:
    DOIT_CONFIG = json.load(file)

# Import all tasks specified in the JSON file marked as true
for task, value in DOIT_CONFIG:
    if value:
        importlib.import_module(task + "_tasks.py", "tasks")
    