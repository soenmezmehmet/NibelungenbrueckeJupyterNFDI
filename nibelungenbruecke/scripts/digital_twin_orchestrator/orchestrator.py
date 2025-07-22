import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
        self.plot_virtual_sensors = {}
        self.plot_model_typ = ""
        
    def _digital_twin_initializer(self):
        """
       Initializes the digital twin model.
       
       Returns:
           DigitalTwin: An instance of the DigitalTwin class initialized with the given parameters.
       
        """
        return DigitalTwin(self.model_parameters_path, self.model_to_run)
        
    def predict_dt(self, digital_twin, input_value, model_to_run, api_key):   ##TODO: Input parameters to be deleted!!
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
                'thermal_h5py_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/Nibelungenbrücke_thermal.h5',
                'thermal_xmdf_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/Nibelungenbrücke_thermal.xmdf',
                'displacement_h5py_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/displacements.h5',
                'displacement_xmdf_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview/displacements.xmdf',
                }

    def compare(self, output, input_value):
        self.updated = (output == 2 * input_value)
        
    def set_api_key(self, key):
        self.api_key = key

    def load(self, simulation_parameters):
        """
        Validates simulation parameters by checking if virtual sensor positions lie within the mesh domain.

        Args:
            simulation_parameters (dict): The simulation parameters including virtual sensor positions.

        Raises:
            ValueError: If any virtual sensor lies outside the mesh domain.
        """
        
        model = simulation_parameters.get('model')
        if model == 'TransientThermal_1':
            path = self.default_parameters['thermal_h5py_path']
        elif model == 'displacement_1':
            path = self.default_parameters['displacement_h5py_path']
        else:
            raise ValueError(f"Unsupported model type: {model}")

        with h5py.File(path, 'r') as f:
            geometry = f['/Mesh/mesh/geometry'][:]

        for sensor in simulation_parameters.get('virtual_sensor_positions', []):
            coords = np.array([sensor['x'], sensor['y'], sensor['z']])

            distances = np.linalg.norm(geometry - coords, axis=1)
            min_dist = np.min(distances)


            threshold = 1.29  ##TODO: Maximum element size is 1.283 m. Outer virtual sensor coordinates smaller than threshold considered in the domain!!

            if min_dist > threshold:
                raise ValueError(
                    f"Virtual sensor '{sensor['name']}' at {coords.tolist()} "
                    f"is outside the mesh domain (nearest node distance: {min_dist:.6f} m)."
                )

        print("All virtual sensors are within the mesh domain.")
       

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
          
        self.prediction = self.predict_dt(self.digital_twin_model, self.simulation_parameters['parameter_update'], self.simulation_parameters['model'], self.api_key)
        
        if not self.prediction:
            pass
        
        else:
            self.plot_virtual_sensors, self.plot_model_typ = self.extract_virtual_sensor_data()
        

    def extract_virtual_sensor_data(self):
        print("\n--- Extracting virtual sensor data ---")
        virtual_sensors = {}
    
        # Determine model and HDF5 path
        model = self.simulation_parameters.get('model')
        if model == 'TransientThermal_1':
            model_typ = "thermal"
            path = self.default_parameters['thermal_h5py_path']
        elif model == 'displacement_1':
            model_typ = "displacement"
            path = self.default_parameters['displacement']
        else:
            raise ValueError(f"Unsupported model type: {model}")
    
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
    
                data_over_time = {}
                for time_str in data_group.keys():
                    time = int(time_str)
                    value = data_group[time_str][nearest_node_idx]
                    if model_typ == "displacement":
                        value = np.linalg.norm(value)  # vector magnitude
                    else:
                        value = value[0]  # scalar temperature
                    data_over_time[time] = value
    
                virtual_sensors[name] = {
                    'coordinates': nearest_node_coord.tolist(),
                    'data': dict(sorted(data_over_time.items()))
                }
    
        return virtual_sensors, model_typ
        
    def plot_virtual_sensor_data(self):
        
        for sensor_name, info in self.plot_virtual_sensors.items():
            data = info['data']
            
            #%%
            start_time_str = self.simulation_parameters['start_time']
            time_step_str = self.simulation_parameters['time_step']
            
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
    
            unit = time_step_str[-3:]
            value = int(time_step_str[:-3])
            if "min" in unit:
                delta = timedelta(minutes=value)
            elif "hou" in unit:
                delta = timedelta(hours=value)
            elif "day" in unit:
                delta = timedelta(days=value)
            else:
                raise ValueError("Unsupported time step format.")
            
            times = [start_time + i * delta for i in range(len(data))]
            
            #%%
            
            values = list(data.values())
    
            plt.figure(figsize=(8, 4))
            plt.plot(times, values, marker='o')
    
            if self.plot_model_typ == "thermal":
                plt.title(f"Temperature at Virtual Sensor: {sensor_name}")
                plt.ylabel("Temperature (°C)")
            elif self.plot_model_typ == "displacement":
                plt.title(f"Displacement Magnitude at Virtual Sensor: {sensor_name}")
                plt.ylabel("Displacement (m)")
    
            plt.xlabel("Time")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            #%%
            self.plot_full_field_response(self.simulation_parameters['full_field_results'])
            
            #%%
            
    ##TODO: 
        
    def plot_full_field_response(self, full_field=True):
        '''
        currently returns the simulation result paths
        '''
        if full_field:
            model = self.simulation_parameters.get('model')
            
            if 'TransientThermal' in model:
                h5py_path = self.default_parameters['thermal_h5py_path']
                xmdf_path = self.default_parameters['thermal_xmdf_path']
                
            elif 'displacement' in model:
                h5py_path = self.default_parameters['displacement_h5py_path']
                xmdf_path = self.default_parameters['displacement_xmdf_path']
                
            print(f"Path to full-field results:")
            print(f"{model} -> h5py_path: {h5py_path}")
            print(f"{model} -> xmdf_path: {xmdf_path}")
                    
                
       #%%

if __name__ == "__main__":
    
    simulation_parameters = {       ##Throw an error checking UQ!!
        'simulation_name': 'TestSimulation',
        'model': 'TransientThermal_1',
        'start_time': '2023-08-11T08:00:00Z',
        'end_time': '2023-12-11T09:01:00Z',
        'time_step': '10min',
        'parameter_update': {'rho': 2800, 'E': 300000000000},       ##TODO: !!
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
    
    orchestrator.plot_virtual_sensor_data()

    ## 

    virtual_sensor_positions = [
        {'x': 1.78, 'y': 0.0, 'z': 26.91, 'name': 'Sensor1'},
        {'x': -1.83, 'y': 0.0, 'z': 0.0, 'name': 'Sensor2'}
        # Note: the real sensor positions are added automatcally by the interface, so you don't need to specify them here.
    ]

    orchestrator.simulation_parameters["virtual_sensor_positions"] = virtual_sensor_positions


    orchestrator.plot_virtual_sensor_data()

    

    #orchestrator.run(simulation_parameters)