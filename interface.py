import os
print(os.getcwd())


import os
import sys

original_cwd = os.getcwd()
root_dir = os.getcwd()
orchestrator_dir = os.path.join(root_dir, 'nibelungenbruecke', 'scripts', 'digital_twin_orchestrator')
os.chdir(orchestrator_dir)
sys.path.insert(0, root_dir)

from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

simulation_parameters = {
    'simulation_name': 'TestSimulation',
    'model': 'TransientThermal_1',
    'start_time': '2023-08-11T08:00:00Z',
    'end_time': '2023-09-11T08:01:00Z',
    'time_step': '10min',
    'virtual_sensor_positions': [
        {'x': 0.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor1'},
        {'x': 1.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'},
        {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor3'},
        {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor4'}
        # Note: the real sensor positions are added automatically by the interface, so you don't need to specify them here.
    ],
    'full_field_results': False, # Set to True if you want full field results, the simulation will take longer and the results will be larger.
    'uncertainty_quantification': False, # Set to True if you want uncertainty quantification, the simulation will take longer and the results will be larger.
}

orchestrator = Orchestrator(simulation_parameters)

key=input("\nEnter the code to connect API: ").strip()
orchestrator.set_api_key(key)

orchestrator.load(simulation_parameters) # Here we first load and then run, so that we can check the inputs before running the simulation and throw an error if something is wrong.
results = orchestrator.run() # The plotting should be separated from the run, so that we can run the simulation without plotting if we want to.


orchestrator.plot_results_at_virtual_sensors()

# Update parameters and reload
new_parameters = simulation_parameters.copy()
new_parameters['parameter_update'] = {'rho': 2900, 'E': 290000000000}

orchestrator.load(new_parameters)
result2 = orchestrator.run()
print("Second run result:", result2)

virtual_sensor_positions = [
        {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor1'},
        {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
        # Note: the real sensor positions are added automatcally by the interface, so you don't need to specify them here.
    ]

orchestrator.simulation_parameters["virtual_sensor_positions"] = virtual_sensor_positions


orchestrator.plot_results_at_virtual_sensors()

print(os.getcwd())