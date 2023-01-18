import json

class GeneratorModel:
    ''' Base class for a generator of synthetic data from a model.'''

    def __init__(self, sensor_positions_path: str, model_parameters: dict):
        try:
            with open(sensor_positions_path) as f:
                self.sensor_positions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {sensor_positions_path} was not found")
        except json.decoder.JSONDecodeError:
            raise json.decoder.JSONDecodeError(f"The file {sensor_positions_path} is not a valid json")
        
        self.model_parameters = model_parameters

    def Generate(self):
        ''' Generate the data from the start'''
        self.LoadGeometry()
        self.GenerateModel()
        self.GenerateData()

    def LoadGeometry(self):
        ''' Load the meshed geometry from a .msh file'''
        raise NotImplementedError("LoadModel should be implemented")

    def GenerateModel(self):
        ''' Generate the FEM model.'''
        raise NotImplementedError("GenerateModel should be implemented")

    def GenerateData(self):
        ''' Run the FEM model and generate the data'''
        raise NotImplementedError("GenerateData should be implemented")