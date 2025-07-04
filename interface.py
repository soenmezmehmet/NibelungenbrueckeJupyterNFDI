import os
print(os.getcwd())


import sys
orchestrator_dir = os.path.join(os.path.dirname(__file__), 'nibelungenbruecke', 'scripts', 'digital_twin_orchestrator')
os.chdir(orchestrator_dir)

sys.path.insert(0, os.path.abspath(os.path.join(orchestrator_dir, '../../../..')))

# Now you can import and run
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

simulation_parameters = {
    'simulation_name': 'TestSimulation',
    'model': 'TransientThermal_1',
    'start_time': '2023-08-11T08:00:00Z',
    'end_time': '2023-09-11T08:01:00Z',
    'time_step': '10min',
    'model_parameter_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json',
    'virtual_sensor_positions': [
        {'x': 0.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor1'},
        {'x': 1.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
        # Note: the real sensor positions are added automatically by the interface, so you don't need to specify them here.
    ],
    'parameter_update': {'rho': 2750, 'E': 310000000000},
    'full_field_results': False, # Set to True if you want full field results, the simulation will take longer and the results will be larger.
    'uncertainty_quantification': False, # Set to True if you want uncertainty quantification, the simulation will take longer and the results will be larger.
}


orchestrator = Orchestrator(simulation_parameters)


key=input("\nEnter the code to connect API: ").strip()
orchestrator.set_api_key(key)

results = orchestrator.run()

print(os.getcwd())