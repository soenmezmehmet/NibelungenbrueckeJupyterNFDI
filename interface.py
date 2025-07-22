import os
import sys

# Save the original working directory
original_cwd = os.getcwd()

# Define project root and orchestrator directory
root_dir = original_cwd
orchestrator_dir = os.path.join(root_dir, 'nibelungenbruecke', 'scripts', 'digital_twin_orchestrator')

# Change to orchestrator directory
os.chdir(orchestrator_dir)

# Add root to sys.path
sys.path.insert(0, root_dir)

# Import Orchestrator
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator


simulation_parameters = {
    'simulation_name': 'TestSimulation',
    'model': 'TransientThermal_1',
    'start_time': '2023-08-11T08:00:00Z',
    'end_time': '2023-09-11T08:01:00Z',
    'time_step': '10min',
    'parameter_update': {'rho': 2800, 'E': 300000000000},       ##TODO: !!
    'virtual_sensor_positions': [
        {'x': 0.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor1'},
        {'x': 1.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'},
        {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor3'},
        {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor4'}
    ],
    'full_field_results': False,
    'uncertainty_quantification': False
}

orchestrator = Orchestrator(simulation_parameters)



key = input("Enter the code to connect API: ").strip()
if not key:
    raise ValueError("API key is required.")
orchestrator.set_api_key(key)

orchestrator.load(simulation_parameters)
results = orchestrator.run()

orchestrator.plot_virtual_sensor_data()


from copy import deepcopy

new_parameters = deepcopy(simulation_parameters)
new_parameters['parameter_update'] = {'rho': 2800, 'E': 290000000000}

orchestrator.load(new_parameters)
result2 = orchestrator.run()
orchestrator.plot_virtual_sensor_data()




new_parameters = deepcopy(orchestrator.simulation_parameters)
new_parameters['virtual_sensor_positions'] = [
    {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor1'},
    {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
]

orchestrator.load(new_parameters)

result3 = orchestrator.run()
print("Third run result:", result3)
orchestrator.plot_virtual_sensor_data()


new_parameters['parameter_update'] = {'rho': 3000, 'E': 290000000000}
result4 = orchestrator.run(new_parameters)
print("Fourth run result:", result4)
orchestrator.plot_virtual_sensor_data()


result5 = orchestrator.run(new_parameters)
print("Fifth run result:", result5)
orchestrator.plot_virtual_sensor_data()

os.chdir(original_cwd)
print("Working directory restored to:", original_cwd)



