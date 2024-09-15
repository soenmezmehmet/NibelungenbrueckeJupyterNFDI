from nibelungenbruecke.scripts.digital_twin_orchestrator.base_model import BaseModel
#from base_model import BaseModel
import dolfinx as df
import json
from nibelungenbruecke.scripts.data_generation.nibelungen_experiment import NibelungenExperiment
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.util import ureg
from nibelungenbruecke.scripts.utilities.API_sensor_retrieval import API_Request, MetadataSaver, Translator
from nibelungenbruecke.scripts.utilities.loaders import load_sensors
from nibelungenbruecke.scripts.utilities.offloaders import offload_sensors
import importlib
import time

    
import ufl
import numpy as np
from copy import deepcopy

from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh
#import dolfinx as df
from mpi4py import MPI
#from nibelungenbruecke.scripts.data_generation.generator_model_base_class import GeneratorModel
from nibelungenbruecke.scripts.utilities.boundary_condition_factory import boundary_condition_factory
from nibelungenbruecke.scripts.utilities.loaders import load_sensors
from nibelungenbruecke.scripts.utilities.offloaders import offload_sensors

class DisplacementModel(BaseModel):
    
    def __init__(self, model_path: str, model_parameters: dict, dt_path: str):
        super().__init__(model_path, model_parameters)
        
        self.material_parameters = self.model_parameters["material_parameters"]
        self.default_p = self._get_default_parameters()
        self.dt_path = dt_path
        self.vs_path = self.model_parameters["material_parameters"]
        
        self.calculate_lame_constants()

        self.dt = model_parameters["dt"]

        # Initialize the load
        self.initial_position = self.model_parameters[
            "initial_position"
        ]  # Initial position of the front left wheel of the vehicle
        self.current_position = deepcopy(
            self.initial_position
        )  # Current position of the front left wheel of the vehicle
        self.historic_position = [
            self.current_position[2]
        ]  # Historic X position of the front left wheel of the vehicle
        self.speed = self.model_parameters["speed"]  # Speed of the vehicle
        self.length_vehicle = self.model_parameters["length"]  # Length of the vehicle
        self.width_vehicle = self.model_parameters["width"]  # Width of the vehicle
        self.load_value = (
            self.model_parameters["mass"] * self.model_parameters["g"] / (self.length_vehicle * self.width_vehicle)
        )  # Load of the vehicle per surface unit
        self.length_road = self.model_parameters["length_road"]  # Length of the road
        self.width_road = self.model_parameters["width_road"]  # Width of the road

        assert (
            self.length_vehicle < self.length_road
        ), "The length of the vehicle is bigger than the length of the road"
        assert self.width_vehicle < self.width_road, "The width of the vehicle is bigger than the width of the road"
        assert (
            self.initial_position[0] > -self.width_road / 2
        ), "The initial position of the vehicle is outside the road width (left)"
        assert (
            self.initial_position[0] + self.width_vehicle < self.width_road / 2
        ), "The initial position of the vehicle is outside the road width (right)"
     
    def LoadGeometry(self):
        pass
    
    def GenerateModel(self):
        self.experiment = NibelungenExperiment(self.model_path, self.material_parameters)
        self.default_p.update(self.experiment.default_parameters())
        self.problem = LinearElasticity(self.experiment, self.default_p)
        
        ###
        
        """Generate the model for the line test load generator."""
        # Generate function space
        self.V = fem.VectorFunctionSpace(self.mesh, ("CG", 1))

        # Load boundary conditions
        self.LoadBCs()

        if self.model_parameters["tension_z"] != 0.0:
            T = fem.Constant(self.mesh, ScalarType((0, 0, self.model_parameters["tension_z"])))
            ds = ufl.Measure("ds", domain=self.mesh)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        f = fem.Constant(self.mesh, ScalarType((0, -self.load_value, 0)))
        f_weight = fem.Constant(
            self.mesh, ScalarType(np.array([0, -self.material_parameters["rho"] * self.model_parameters["g"], 0]))
        )
        self.evaluate_load()
        self.a = ufl.inner(self.sigma(u), self.epsilon(v)) * ufl.dx
        if self.model_parameters["tension_z"] != 0.0:
            self.L = ufl.dot(f, v) * self.ds_load(1) + ufl.dot(f_weight, v) * ufl.dx + ufl.dot(T, v) * ds
        else:
            self.L = ufl.dot(f, v) * self.ds_load(1) + ufl.dot(f_weight, v) * ufl.dx
        
    def GenerateData(self):
        """Generate data based on the model parameters."""

        self.api_request = API_Request()
        self.api_dataFrame = self.api_request.fetch_data()

        metadata_saver = MetadataSaver(self.model_parameters, self.api_dataFrame)
        metadata_saver.saving_metadata()

        translator = Translator(self.model_parameters)
        translator.translator_to_sensor()

        self.problem.import_sensors_from_metadata(self.model_parameters["MKP_meta_output_path"])
        self.problem.fields.temperature = self.problem.fields.displacement #!!
        self.problem.solve()

        translator.save_to_MKP(self.api_dataFrame)
        translator.save_virtual_sensor(self.problem)

        if self.model_parameters["paraview_output"]:
            with df.io.XDMFFile(self.problem.mesh.comm, self.model_parameters["paraview_output_path"]+"/"+self.model_parameters["model_name"]+".xdmf", "w") as xdmf:
                xdmf.write_mesh(self.problem.mesh)
                xdmf.write_function(self.problem.fields.displacement)
                
        #####
        
        """Generate the displacement data for the line test load generator."""
        # Code to generate displacement data

        if self.model_parameters["paraview_output"]:
            pv_file = df.io.XDMFFile(
                self.mesh.comm,
                self.model_parameters["paraview_output_path"] + "/" + self.model_parameters["model_name"] + ".xdmf",
                "w",
            )

        # Solve the problem
        sensors = load_sensors(self.sensor_positions)
        i = 0
        converged = False
        while not converged:
            self.GenerateModel()
            problem = fem.petsc.LinearProblem(
                self.a, self.L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
            )
            self.displacement = problem.solve()
            self.displacement.name = "Displacement"

            # Sensor measurement (should be adapted with the wrapper)
            for sensor in sensors:
                sensor.measure(self, i * self.dt)

            # Paraview output
            if self.model_parameters["paraview_output"]:
                if i == 0:
                    pv_file.write_mesh(self.mesh)
                pv_file.write_function(self.displacement, i * self.dt)
                L_vector = fem.assemble_vector(fem.form(self.L))
                L_function = fem.Function(self.V, x=L_vector, name="Load")
                pv_file.write_function(L_function, i * self.dt)
                pv_file.close()
                # Store the value at the sensors

            converged = self.advance_load(self.dt)
            i += 1

        offload_sensors(
            sensors,
            self.output_parameters["output_path"] + "/" + self.model_parameters["model_name"],
            self.output_parameters["output_format"],
        )

        print(f"Number of iterations: {i}")
        
    def LoadBCs(self):
        """Load the boundary conditions for the line test load generator."""
        bcs = []
        for bc_name, bc_model in self.model_parameters["boundary_conditions"].items():
            bc = boundary_condition_factory(self.mesh, bc_model["model"], self.V, bc_model)
            bcs.append(bc)

        self.bcs = bcs
        
    def epsilon(self, u):
        """Compute the strain tensor for the line test load generator."""
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(self, u):
        """Compute the stress tensor for the line test load generator."""
        return self.material_parameters["lambda"] * ufl.nabla_div(u) * ufl.Identity(
            len(u)
        ) + 2 * self.material_parameters["mu"] * self.epsilon(u)

    def calculate_lame_constants(self):
        """Calculate the Lame constants for the line test load generator."""
        E_modulus = self.material_parameters["E"]
        nu = self.material_parameters["nu"]
        self.material_parameters["lambda"] = (E_modulus * nu) / ((1 + nu) * (1 - 2 * nu))
        self.material_parameters["mu"] = E_modulus / (2 * (1 + nu))

    def advance_load(self, dt):
        """Advance the load for the line test load generator."""
        self.current_position[2] += self.speed * dt
        self.historic_position.append(self.current_position[2])
        # self.evaluate_load()

        return self.current_position[2] > self.length_road + self.length_vehicle

    def evaluate_load(self):
        """Evaluate the load for the line test load generator."""

        # Apply local load (surface force) in subdomain
        load_subdomain = LoadSubDomain(
            corner=self.current_position, length=self.length_vehicle, width=self.width_vehicle
        )
        subdomain = mesh.locate_entities(self.mesh, self.mesh.topology.dim - 1, marker=load_subdomain.inside)
        subdomain_values = np.full_like(subdomain, 1)
        facet_tags = mesh.meshtags(self.mesh, self.mesh.topology.dim - 1, subdomain, subdomain_values)
        self.ds_load = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)

    @staticmethod
    def _get_default_parameters():
        """Get the default parameters for the line test load generator."""
        default_parameters = {
            "model_name": "displacements",
            "paraview_output": False,
            "paraview_output_path": "output/paraview",
            "material_parameters": {"rho": 1.0, "mu": 1, "lambda": 1.25},
            "tension_z": 0.0,
            "mass": 1000,
            "g": 9.81,
            "initial_position": [0.0, 0.0, 0.0],
            "speed": 1.0,
            "length": 1.0,
            "width": 1.0,
            "length_road": 10.0,
            "width_road": 10.0,
            "dt": 1.0,
            "boundary_conditions": [{"model": "clamped_boundary", "side_coord": 0.0, "coord": 0}],
        }

        return default_parameters
        
    @staticmethod
    def _get_default_parameters():
        """
        Get default material parameters.

        Returns:
            dict: Default material parameters.
        """
        default_parameters = {
            "rho":7750 * ureg("kg/m^3"),
            "E":210e9 * ureg("N/m^2"),
            "nu":0.28 * ureg("")
        }
        return default_parameters
    
    def update_input(self, sensor_input):
        
        with open(self.dt_path, 'r') as f:
            dt_params = json.load(f)
        
        # currently, only updates E value
        #TODO: Make this part more automated/flexible!  
        if isinstance(sensor_input, (int, float)):
            dt_params[0]["parameters"]["E"] = sensor_input
            
            with open(self.dt_path, 'w') as file:
                json.dump(dt_params, file, indent=4)
            return True
        else:
            return False
        
    def solve(self):

        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()
        
        #TODO: API Request output error!!
        self.sensor_out = self.api_dataFrame['E_plus_080DU_HSN-u-_Avg1'].iloc[-1] # *1000 #Convertion from meter to milimeter
        #self.sensor_out = self.api_dataFrame['E_plus_413TU_HSS-m-_Avg1'].iloc[-1]
   
        
        vs_file_path = self.model_parameters["virtual_sensor_added_output_path"]
        with open(vs_file_path, 'r') as file:
            self.vs_data = json.load(file)        
        
        #TODO: API Request output error!!
        #self.vs_sensor_out = self.vs_data['virtual_sensors']['E_plus_413TU_HSS-m-_Avg1']['displacements'][-1][0]
        self.vs_sensor_out = self.vs_data['virtual_sensors']['E_plus_080DU_HSN-u-_Avg1']['displacements'][-1][0]
        
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
    

# Define subdomain where the load should be applied
class LoadSubDomain:
    def __init__(self, corner, length, width):
        self.corner = corner
        self.length = length
        self.width = width

    def inside(self, x):
        return np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.logical_and(x[0] > self.corner[0], x[0] < self.corner[0] + self.width), x[2] < self.corner[2]
                ),
                x[2] > self.corner[2] - self.length,
            ),
            np.isclose(x[1], self.corner[1]),
        )

 