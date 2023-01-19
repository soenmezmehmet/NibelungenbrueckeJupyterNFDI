import ufl

from petsc4py.PETSc import ScalarType

from dolfinx import fem

from data_generation.generator_model_base_class import GeneratorModel
from utilities.boundary_condition_factory import boundary_condition_factory

class DisplacementGenerator(GeneratorModel):
    ''' Generates the displacements at the sensors for a given load configuration.'''
    
    def __init__(self, model_path: str, sensor_positions_path: str, model_parameters: dict, output_parameters: dict):
        super().__init__(model_path, sensor_positions_path, model_parameters, output_parameters)

        self.material_parameters = model_parameters["material_parameters"]


    def GenerateModel(self):
        # Code to generate the displacement model
        
        # Generate function space
        self.V = fem.VectorFunctionSpace(self.mesh, ("CG", 1))

        # Load boundary conditions
        self.LoadBCs()

        T = fem.Constant(self.mesh, ScalarType((0, 0, self.model_parameters["tension_z"])))
        ds = ufl.Measure("ds", domain=self.mesh)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        f = fem.Constant(self.mesh, ScalarType((0, -self.material_parameters["rho"]*self.material_parameters["g"],0)))
        self.a = ufl.inner(self.sigma(u), self.epsilon(v)) * ufl.dx
        self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    @GeneratorModel.sensor_offloader_wrapper
    def GenerateData(self):
        # Code to generate displacement data

        # Solve the problem
        problem = fem.petsc.LinearProblem(self.a, self.L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.displacement = problem.solve()
       
        # The wrapper takes care of the offloading


    def LoadBCs(self):

        bcs = []
        for bc_model in self.model_parameters["boundary_conditions"]:
            bc = boundary_condition_factory(self.mesh,bc_model["model"],self.V, bc_model)
            bcs.append(bc)

        self.bcs = bcs

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(self, u):
        return self.material_parameters["lambda"] * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*self.material_parameters["mu"]*self.epsilon(u)

    def _get_default_parameters():
        default_parameters = {
            "model_name":"displacements",
            "material_parameters":{
                "rho": 1.0,
                "g": 100,
                "mu": 1,
                "lambda": 1.25
            },
            "tension_z": 0.0,
            "boundary_conditions": [{
                "model":"clamped_boundary",
                "side_coord": 0.0,
                "coord": 0
            }]
        }

        return default_parameters
