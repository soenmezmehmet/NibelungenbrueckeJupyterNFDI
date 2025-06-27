import json
import time
import pickle
import importlib
import numpy as np
from tqdm import tqdm
import dolfinx as df
import matplotlib.pyplot as plt

from fenicsxconcrete.util import ureg
#from fenicsxconcrete.finite_element_problem.thermomechanical_nibelungenbruecke_demonstrator import ThermoMechanicalNibelungenBrueckeProblem
from .thermomechanical_nibelungenbruecke_demonstrator import ThermoMechanicalNibelungenBrueckeProblem

from nibelungenbruecke.scripts.utilities.loaders import load_sensors
from nibelungenbruecke.scripts.utilities.offloaders import offload_sensors
from nibelungenbruecke.scripts.digital_twin_orchestrator.base_model import BaseModel
from nibelungenbruecke.scripts.data_generation.nibelungen_experiment import NibelungenExperiment
from nibelungenbruecke.scripts.utilities.API_sensor_retrieval import API_Request, MetadataSaver, Translator


class ThermalModel(BaseModel):
    """
    A class representing a thermal model for NB simulations.
    
    """
    
    def __init__(self, model_path: str, model_parameters: dict, dt_path: str):
        """
        Initializes the ThermalModel with the given paths and parameters.
        
        Args:
            model_path (str): Path to the model geometry (mesh) file.
            model_parameters (dict): Dictionary containing material properties 
            and model-specific parameters.
            dt_path (str): Path to the digital twin parameter file (JSON format).
        """
        super().__init__(model_path, model_parameters)
        
        self.dt_path = dt_path
        self.vs_path = self.model_parameters["virtual_sensor_added_output_path"] ##TODO: !!
        
    def LoadGeometry(self):
        """
        Loads the geometry of the model for simulation.
        
        """
        pass
    
    def GenerateModel(self):
        """
        Generates the model by setting up the attributes experiment and problem with their parameters.
        
        This includes setting up experiment and problem from material parameters and related data 
        using the 'NibelungenExperiment' and `LinearElasticityNibelungenbrueckeDemonstrator`respectively.
        
        """
        self.experiment = NibelungenExperiment(self.model_path, self.model_parameters)
        self.problem = ThermoMechanicalNibelungenBrueckeProblem(
            [self.GenerateData, self.PostAPIData, self.ParaviewProcess], self.experiment, self.experiment.parameters, pv_path=self.model_parameters["paraview_output_path"])
        
    def GenerateData(self):
        """
        Requests data from an API, transforms it into metadata, 
        and saves it for use with virtual sensors.

        """

        self.api_request = API_Request(self.model_parameters["secret_path"])
        self.api_dataFrame = self.api_request.fetch_data()

        metadata_saver = MetadataSaver(self.model_parameters, self.api_dataFrame)
        metadata_saver.saving_metadata()    ##TODO:

        self.translator = Translator(self.model_parameters)
        self.translator.translator_to_sensor(self.experiment.mesh)
        
        self.problem.import_sensors_from_metadata(
            self.model_parameters["MKP_meta_output_path"])
        
    def SolveMethod(self):
        """
       Solves the model
       """
        #self.problem.dynamic_solve()        ##TODO: change the name!
        
        #time = 0
        #while time < 190:
            # problem update_and_solve()
        #    self.problem.solve()
        #    time += 1
        #    tqdm.write(f"Progress: {time}/190")
        
        database = {
            "real_sensor_data": {},
            "virtual_sensor_data": {}
        }

        data = self.api_dataFrame

        air_temperature_array = np.zeros(len(data))
        inner_temperature_array = np.zeros(len(data))
        i=0
        for _, data_point in tqdm(data.iterrows(), total=len(data)):
            air_temperature_array[i] = data_point["F_plus_000TA_KaS-o-_Avg1"] + 273.15
            inner_temperature_array[i] = data_point["E_plus_040TI_HSS-u-_Avg"] + 273.15 
            new_parameters = {
                "air_temperature": air_temperature_array[i] ,
                "inner_temperature": inner_temperature_array[i],
                "shortwave_irradiation": data_point["F_plus_000S_KaS-o-_Avg1"],
                "calculate_shortwave_irradiation": False,
            }
            self.problem.update_parameters(new_parameters)
            # print(problem.air_temperature.value)
            self.problem.solve()
            i+=1
            
            #%%
            
            for sensor_id in data.columns:
                if sensor_id in ["F_plus_000TA_KaS-o-_Avg1", "E_plus_040TI_HSS-u-_Avg", 'F_plus_000TA_KaS-o-_Avg1', 'F_plus_000S_KaS-o-_Avg1',
                       'E_plus_040TU_HS--u-_Avg1', 'E_plus_040TI_HSS-u-_Avg']:      ##TODO: Should be extended to consist all the temperature sensors
                    database["real_sensor_data"].setdefault(sensor_id, [])
                    database["virtual_sensor_data"].setdefault(sensor_id, [])
                    
                    database["real_sensor_data"][sensor_id].append(data_point[sensor_id])
    
                    # Append virtual sensor data
                    temperature_value = self.problem.sensors.get(sensor_id, None)
                    if temperature_value is not None:
                        temperature_value_list = temperature_value.data[-1].tolist()
                        database["virtual_sensor_data"][sensor_id].append(temperature_value_list)
                
            #%%
         
            if i == 10:
                break
            #%%
        self.plot_all_sensors_together(database)
            
    def plot_all_sensors_together(self, database):
        real_data = database["real_sensor_data"]
        virtual_data = database["virtual_sensor_data"]
    
        plt.figure(figsize=(12, 6))
        for sensor_id in real_data:
            # Real sensor data
            real_values = real_data[sensor_id]
    
            # Virtual sensor data (flatten lists)
            virtual_values = [v[0] if isinstance(v, list) else v for v in virtual_data[sensor_id]]
    
            # Plot real and virtual on same graph with different styles
            plt.plot(real_values, label=f"{sensor_id} - Real", linestyle='-')
            plt.plot(virtual_values, label=f"{sensor_id} - Virtual", linestyle='--')
    
        plt.title("Sensor Data: Real vs Virtual (All Sensors)")
        plt.xlabel("Timestep")
        plt.ylabel("Sensor Value")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

 

#%%
    def PostAPIData(self):
        """
        Saving data of virtual sensor' after the model solve
        """
        
        self.translator.save_to_MKP(self.api_dataFrame)
        self.translator.save_virtual_sensor(self.problem)        
        
    def ParaviewFirstRun(self):
        """
        Handles the first run of Paraview data output.
        
        This method is useful for initializing the Paraview output for the 
        first time, setting up the model visualization,
        and preparing the mesh for further data writing in later iterations.
        """
        if self.model_parameters["paraview_output"]:
            with df.io.XDMFFile(self.problem.mesh.comm, self.model_parameters["paraview_output_path"]
                                +"/"+self.model_parameters["model_name"]+".xdmf", "w") as xdmf:
                xdmf.write_mesh(self.problem.mesh)
                #xdmf.write_function(self.problem.fields.displacement, self.problem.time)     
                
    def ParaviewProcess(self):
        """
        Processes the Paraview data during each timestep.
        It writes the displacement fields to the Paraview XDMF file.
        """
        
        if self.model_parameters["paraview_output"]:
            with df.io.XDMFFile(self.problem.mesh.comm, 
                                self.model_parameters["paraview_output_path"]+
                                "/"+self.model_parameters["model_name"]+".xdmf", "a") as xdmf:
                # xdmf.write_mesh(self.problem.mesh)
                xdmf.write_function(self.problem.fields.displacement, 
                                    self.problem.time)


    def update_parameters(self, updates, target_name=None):
        """
        Updates the specified parameters in the digital twin parameter file 
        (JSON format).
        
        Args:
            updates (dict): A dictionary containing parameters to be updated 
            (key: value).
            target_name (str): The name of the model whose parameters need to 
            be updated (optional).
        

        """
        try:
            with open(self.dt_path, 'r') as f:
                dt_params = json.load(f)
    
            updated = False
            model_type_params = None
            
            # Update parameters in matching entries
            for entry in dt_params:
                if entry["name"] == target_name:
                    for key, value in updates.items():
                        if key in entry["parameters"]:
                            if entry["parameters"][key] != value:
                                entry["parameters"][key] = value
                                self.problem.p[key] = value  ##TODO: problem.p update!!
                                model_type_params = entry
                                updated = True
   
            # Save the updated JSON back to the file
            if updated:
                with open(self.dt_path, 'w') as file:
                    json.dump(dt_params, file, indent=4)
                return True, model_type_params
            else:
                return False, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    
#%%
    def solve(self):
        """
        Reloading, model generating and solving model.
        
        With use of callback() methods, ceratin methods are called in each step 
        of moving load case.
        
        TODO:
            The re-loading of the model can be avoidede by recalculating and 
            modifying the values in the constant fields of the Lamé constants
            

        """
        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()
        self.SolveMethod()
  

    def export_output(self, path: str): #TODO: json_path as a input parameters!! -> Changes' been done!
        #json_path = "output_data.json" #TODO: move to json file
        
        json_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" + path + ".json"
        try:
            with open(json_path, 'r') as file:
                output_data = json.load(file)
                
        except FileNotFoundError:
            output_data = {}
            
        output_data.setdefault('real_sensor_output', []).append(self.sensor_out)
        output_data.setdefault('virtual_sensor_output', []).append(self.vs_sensor_out)

        local_time = time.localtime()
        output_data.setdefault('time', []).append(time.strftime("%y-%m-%d %H:%M:%S", local_time))

        with open(self.dt_path, 'r') as f:
            dt_params = json.load(f)
        output_data.setdefault('Input_parameter', []).append(dt_params[0]["parameters"]["E"])

        with open(json_path, 'w') as file:
            json.dump(output_data, file)
            
        return json_path
    
    def fields_assignment(self, data):
        if data == None:
            pass
        
        else:
            for i in data.keys():
                if i == "displacement":
                    self.problem.fields.displacement = data[i]
                elif i == "temperature":
                    self.problem.fields.temperature = data[i]
        
    def fields_data_storer(self, path):
        data_to_store = {}

        for i in dir(self.problem.fields):
            if not i.startswith("_"):
                k = getattr(self.problem.fields, i)
                if k:
                    data_to_store[i] = k.x.array[:]
        try:
            pkl_path = "../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/" + path
            with open(f"{pkl_path}_params.pkl", "wb") as f:
                pickle.dump(data_to_store, f)
                
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            


#%%
    
if __name__ == "__main__": 
    
    #model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
    model_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh_3d.msh'
    
    model_parameters = {'model_name': 'displacements',
     'df_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_df_output.csv',
     'meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/sensors/API_meta_output.json',
     'MKP_meta_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_meta_output.json',
     'MKP_translated_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/MKP_translated.json',
     'virtual_sensor_added_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/sensors/virtual_sensor_added_translated.json',
     'cache_path': '',
     'paraview_output': True,
     'paraview_output_path': '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/output/paraview',
     #'material_parameters': {'E': 40000000000000.0, 'nu': 0.2, 'rho': 2350},
     'material_parameters': {},
     "secret_path" : "/home/msoenmez/Desktop/API_request_password",
     'thermal_material_properties': {},
     'tension_z': 0.0,
     'mass': 50000.0,
     'g': 9.81,
     'initial_position': [0.0, 0.0, 0.0],
     'speed': 1.0,
     'length': 7.5,
     'width': 2.5,
     'height': 6.5,
     'length_road': 95.0,
     'width_road': 14.0,
     'thickness_deck': 0.2,
     'dt': 30.0,
     'reference_temperature': 300,
     'temperature_coefficient': 1e-05,
     'temperature_alpha': 1e-05,
     'temperature_difference': 5.0,
     'reference_height': -2.5,
     'boundary_conditions': {'bc1': {'model': 'clamped_edge',
       'side_coord_1': 0.0,
       'coord_1': 2,
       'side_coord_2': 0.0,
       'coord_2': 1},
      'bc2': {'model': 'clamped_edge',
       'side_coord_1': 95.185,
       'coord_1': 2,
       'side_coord_2': 0.0,
       'coord_2': 1}}}
    
    dt_path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_parameters.json'
    
    
    dm = ThermalModel(model_path, model_parameters, dt_path)
    dm.solve()
