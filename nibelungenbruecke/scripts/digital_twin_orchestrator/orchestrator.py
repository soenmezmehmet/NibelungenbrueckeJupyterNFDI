import json
import numpy as np
import h5py
import matplotlib.pyplot as plt

from nibelungenbruecke.scripts.digital_twin_orchestrator.digital_twin import DigitalTwin
from nibelungenbruecke.scripts.utilities.mesh_point_detector import query_point

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

    def __init__(self, simulation_parameters):    
        """
        Initializes the Orchestrator.
        
        Args:
            model_parameters_path (str): Path to the model parameters dictionary.
            model_to_run (str): Specifies which predefined model to execute. Defaults to "Displacement_1".
        
        """

        self.updated = False
        self.simulation_parameters = simulation_parameters
        self.model_to_run = simulation_parameters['model']

        self.default_parameters = self.default_parameters()
        self.model_parameters_path = self.default_parameters['model_parameter_path']
        
        self.digital_twin_model = self._digital_twin_initializer()
        
    def _digital_twin_initializer(self):
        """
       Initializes the digital twin model.
       
       Returns:
           DigitalTwin: An instance of the DigitalTwin class initialized with the given parameters.
       
        """
        return DigitalTwin(self.model_parameters_path, self.model_to_run)
        
    def predict_dt(self, digital_twin, input_value, model_to_run, api_key):   
        """
        Runs "prediction" method of specified digital twin object.
        
        Args:
            digital_twin (DigitalTwin): The digital twin model instance.
            input_value : The input data for prediction.
            model_to_run (str): Specifies which predefined model to execute.
        
        """
        return digital_twin.predict(input_value, model_to_run, api_key)
    
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
    

    ##TODO: Include time steps 
    def default_parameters(self):
    
        import random
        def generate_random_parameters():
            """
            Generates 'rho' and 'E' values.
            """
            params: dict={}
            random_value_rho = random.randint(90 // 5, 160 // 5) * 100
            random_value_E = random.randint(100 // 5, 225 // 5) * 10**10
            params['rho'] = random_value_rho
            params['E'] = random_value_E

            return params
        
        return {'model_parameter_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json',
                'parameters': generate_random_parameters(),
                'thermal_h5py_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/pv_output_full.h5',
                'displacement_h5py_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/displacements.h5',
                        }

    def compare(self, output, input_value):
        self.updated = (output == 2 * input_value)
        
    def set_api_key(self, key):
        self.api_key = key

    def load(self, simulation_parameters):      ##this method should be checking if the virtuals sensor are in the domain of the mesh!!! Wrong implementation
        """
        Loads and validates simulation parameters before running.
        Useful for checking correctness before execution.
        """
        required_keys = ['model', 'start_time', 'end_time']     ##TODO: 
        missing = [key for key in required_keys if key not in simulation_parameters]
        if missing:
            raise ValueError(f"Missing required simulation parameters: {missing}")
        
        self.simulation_parameters = simulation_parameters
        self.model_to_run = simulation_parameters['model']
        self.digital_twin_model = self._digital_twin_initializer()
        
        print("Simulation parameters : \n")
        print(simulation_parameters, "\n")



    def run(self, simulation_parameters=None):
        """
        Runs the digital twin model prediction.
        
        TODO:
        - Implement conditional execution based on prediction type.
        - Support more flexible input types.
        
        Args:
            input_value : The input data for prediction.
            model_to_run (str): Specifies which predefined model to execute.
        
        """

        if simulation_parameters is None:
            simulation_parameters = self.simulation_parameters
            
        else:
            self.simulation_parameters = simulation_parameters
          
        self.prediction = self.predict_dt(self.digital_twin_model, self.default_parameters['parameters'], simulation_parameters['model'], self.api_key)
        

    def plot_results_at_virtual_sensors(self):
        virtual_sensors = {}
    
        # Determine model and HDF5 path
        if self.simulation_parameters['model'] == 'TransientThermal_1':
            model_typ = "thermal"
            path = self.default_parameters['thermal_h5py_path']
        elif self.simulation_parameters['model'] == 'displacement_1':
            model_typ = "displacement"
            path = self.default_parameters['displacement']
    
        with h5py.File(path, 'r') as f:
            geometry = f['/Mesh/mesh/geometry'][:]
            data_group = f['/Function/temperature'] if model_typ == "thermal" else f['/Function/displacement']
    
            for sensor in self.simulation_parameters['virtual_sensor_positions']:
                name = sensor['name']
                coords = np.array([sensor['x'], sensor['y'], sensor['z']])
    
                projected = query_point(coords, self.prediction.problem.mesh)[0]
                distances = np.linalg.norm(geometry - projected, axis=1)
                nearest_node_idx = np.argmin(distances)
                nearest_node_coord = geometry[nearest_node_idx]
    
                print(f"\nSensor '{name}' -> nearest node index: {nearest_node_idx}")
                print(f"Nearest node coordinates: {nearest_node_coord}")
    
                # Collect time series data
                data_over_time = {}
                for time_str in data_group.keys():
                    time = int(time_str)
                    value = data_group[time_str][nearest_node_idx]
                    if model_typ == "displacement":
                        value = np.linalg.norm(value)  # Optional: Use vector magnitude
                    else:
                        value = value[0]  # scalar temperature
                    data_over_time[time] = value
    
                # Save to dict
                virtual_sensors[name] = {
                    'coordinates': nearest_node_coord.tolist(),
                    'data': dict(sorted(data_over_time.items()))
                }
    
        # Plot results
        for sensor_name, info in virtual_sensors.items():
            data = info['data']
            times = list(data.keys())
            values = list(data.values())
    
            plt.figure(figsize=(8, 4))
            plt.plot(times, values, marker='o')
    
            if model_typ == "thermal":
                plt.title(f"Temperature at Virtual Sensor: {sensor_name}")
                plt.ylabel("Temperature (°C)")
            elif model_typ == "displacement":
                plt.title(f"Displacement Magnitude at Virtual Sensor: {sensor_name}")
                plt.ylabel("Displacement (m)")  # or mm, depending on units
    
            plt.xlabel("Time")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            
    def plot_results_at_real_sensors(self):
        print("Virtual sensors to be plotted with the corresponding real sensor")
        pass

                            
            
        

if __name__ == "__main__":
    
    simulation_parameters = {
        'simulation_name': 'TestSimulation',
        'model': 'TransientThermal_1',
        'start_time': '2023-08-11T08:00:00Z',
        'end_time': '2023-12-11T09:01:00Z',
        'time_step': '10min',
        'virtual_sensor_positions': [
            {'x': 0.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor1'},
            {'x': 1.0, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
            # Note: the real sensor positions are added automatically by the interface, so you don't need to specify them here.
        ],
        'full_field_results': False, # Set to True if you want full field results, the simulation will take longer and the results will be larger.
        'uncertainty_quantification': False, # Set to True if you want uncertainty quantification, the simulation will take longer and the results will be larger.
    }


    orchestrator =  Orchestrator(simulation_parameters)
    key = input("\nEnter the code to connect API: ").strip()
    orchestrator.set_api_key(key)
    orchestrator.run()
    
    orchestrator.plot_results_at_virtual_sensors()

    ## 

    virtual_sensor_positions = [
        {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor1'},
        {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
        # Note: the real sensor positions are added automatcally by the interface, so you don't need to specify them here.
    ]

    orchestrator.simulation_parameters["virtual_sensor_positions"] = virtual_sensor_positions


    orchestrator.plot_results_at_virtual_sensors()

    

    orchestrator.run(simulation_parameters)