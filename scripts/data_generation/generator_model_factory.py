import importlib.util

from generator_model_base_class import GeneratorModel

def generator_model_factory(filepath, sensor_positions, model_parameters):
    # Import the module from the filepath
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the derived class of GeneratorModel in the imported module
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, GeneratorModel):
            model_class = obj
            break
    else:
        raise ValueError(f"No derived class of GeneratorModel found in {filepath}")
    
    # Create an instance of the derived class with the given parameters
    return model_class(sensor_positions, model_parameters)