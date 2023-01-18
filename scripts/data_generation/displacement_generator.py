import json
import numpy as np
import ufl

from petsc4py.PETSc import ScalarType
from mpi4py import MPI

from dolfinx import fem, mesh

from generator_model_base_class import GeneratorModel
from utilities.boundary_condition_factory import boundary_condition_factory
from utilities.sensors import *
from utilities.loaders import load_sensors
from utilities.offloaders import offload_sensors

class DisplacementGenerator(GeneratorModel):
    ''' Generates the displacements at the sensors for a given load configuration.'''
    
    def __init__(self, model_path: str, sensor_positions_path: str, model_parameters: dict):
        super().__init__(model_path, sensor_positions_path, model_parameters)

        self.material_parameters = model_parameters["material_parameters"]


    def GenerateModel(self):
        # Code to generate the displacement model
        
        # Generate function space
        self.V = fem.VectorFunctionSpace(self.mesh, ("CG", 1))

        # Load boundary conditions
        self.LoadBCs()

        T = fem.Constant(self.mesh, ScalarType((0, 0, 0)))
        ds = ufl.Measure("ds", domain=self.mesh)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        f = fem.Constant(self.mesh, ScalarType((0, 0, -self.material_parameters["rho"]*self.model_parameters["g"])))
        self.a = ufl.inner(self.sigma(u), self.epsilon(v)) * ufl.dx
        self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    @GeneratorModel.sensor_offloader_wrapper
    def GenerateData(self):
        # Code to generate displacement data

        # Solve the problem
        problem = fem.petsc.LinearProblem(self.a, self.L, bcs=[self.bcs], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.displacement = problem.solve()

        # The wrapper takes care of the offloading


    def LoadBCs(self):

        bcs = []
        for bc_model in self.model_parameters["boundary_conditions"]:
            bc = boundary_condition_factory(self.mesh,self.V, bc_model)
            bcs.append(bc)

        self.bcs = bcs

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(self, u):
        return self.material_parameters["lambda"] * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*self.material_parameters["mu"]*self.epsilon(u)

    def _get_default_parameters():
        default_parameters = {
            "model_path": "input/models/mesh.msh",
            "output_path": "data",
            "output_format": ".h5",
            "generation_models_list": [None],
        }

        return default_parameters
