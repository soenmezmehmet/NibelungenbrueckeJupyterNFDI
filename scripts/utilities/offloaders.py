import h5py
import os
import sys

# Get the parent directory of the current script
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the parent directory to sys.path
sys.path.append(root_path)

def offload_sensors(sensor_list: list, output_path: str, output_format: str):
    ''' Offloads the information of a list of sensors to a file'''
    
    
    if output_format == ".h5":

        with h5py.File(output_path+output_format, "w") as f:
            for sensor, sensor_id in zip(sensor_list,range(len(sensor_list))):
                group = f.create_group(sensor.name+str(int(sensor_id)))
                group.create_dataset("Type", data=sensor.name.encode())
                if hasattr(sensor,"where"):
                    group.create_dataset("Position", data=sensor.where)
                group.create_dataset("Time", data=sensor.time)
                group.create_dataset("Data", data=sensor.data)
                if hasattr(sensor,"max"):
                    group.create_dataset("Max", data=sensor.max)
    else:
        raise NotImplementedError()