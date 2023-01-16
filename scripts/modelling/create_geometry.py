import gmsh
import numpy as np
import os
import sys

# Get the parent directory of the current script
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(root_path)

from utilities.checks import check_path_exists

def create_geometry(parameters):
    "Creates the geometry of the Nibelungenbrücke from a set of parameters and the cross-sections"
    
    # Import parameters
    geo_parameters = _get_default_parameters()
    for key, value in parameters.items():
        geo_parameters[key] = value

    # Sanity checks
    check_path_exists(geo_parameters["cross_section_path"]+"_span"+geo_parameters["cross_section_format"])
    check_path_exists(geo_parameters["cross_section_path"]+"_pilot"+geo_parameters["cross_section_format"])
    
    _check_valid_formats(geo_parameters["output_format"])
    _check_valid_extrusion(geo_parameters["extrude"])

    #### CREATE THE GEOMETRY ####
    
    #  0. Initialize gmsh and parameters
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Create a new model
    gmsh.model.add("bridge")

    # Import cross-section geometries

    # Span
    gmsh.merge(geo_parameters["cross_section_path"]+"_span"+geo_parameters["cross_section_format"])
    points_span = gmsh.model.getEntities(0)
    lines_span = gmsh.model.getEntities(1)

    # Pilot
    gmsh.merge(geo_parameters["cross_section_path"]+"_pilot"+geo_parameters["cross_section_format"])
    points_pilot = [new_point for new_point in gmsh.model.getEntities(0) if not new_point in points_span]
    lines_pilot = [new_line for new_line in gmsh.model.getEntities(1) if not new_line in lines_span]

    ### SAMPLER

    def sampler(n, length, mode):
        x = np.linspace(0, length, n)
        if mode == "linear":
            y = np.zeros(n)
            for i in range(n):
                if x[i] < length/2:
                    y[i] = (x[i]/(length/2))
                else:
                    y[i] = 1-((x[i]-(length/2))/(length/2))
        elif mode == "parabolic":
            y = 1-((x-length/2)/(length/2))**2
        else:
            raise ValueError("Invalid mode. Choose 'linear' or 'parabolic'.")
        return y

    interpolation_coords_x = sampler(geo_parameters["number_of_divisions"], geo_parameters["length"] ,"linear")
    interpolation_coords_y = sampler(geo_parameters["number_of_divisions"], geo_parameters["length"] ,geo_parameters["interpolation"])
    interpolation_coords_z = np.linspace(0,geo_parameters["length"],geo_parameters["number_of_divisions"])

    ## Extrude sections
    if geo_parameters["extrude"] == "lines":
        for line_span, line_pilot in zip(lines_span, lines_pilot):

            # Identify points
            # for point in line
            #     go to interp coords
            #     extrude point into a line
            #     add new point to list of new points
            # join the new points into a line
            # add the line to list of lines to be extruded
            
            lines = []
            start_point_span = gmsh.model.getBoundary(dimTags=line_span)[0]
            x_span, y_span, z_span = gmsh.model.getValue(start_point_span[0],start_point_span[1],[])
            start_point_pilot = gmsh.model.getBoundary(line_pilot)[0]
            x_pilot, y_pilot,z_pilot = gmsh.model.getValue(start_point_pilot[0],start_point_pilot[1],[])
            lines += [line_pilot] 

            # Extrude arrays
            dx = [(x_span-x_pilot)*int_coord for int_coord in np.diff(interpolation_coords_x)]
            dy = [(y_span-y_pilot)*int_coord for int_coord in np.diff(interpolation_coords_y)]
            dz = [int_coord for int_coord in np.diff(interpolation_coords_z)]

            for dx_i, dy_i, dz_i in zip(dx, dy, dz):
                newDimTags = gmsh.model.geo.extrude(lines[-1],dx_i,dy_i,dz_i)
                lines += [newDimTags[0]]
                gmsh.model.geo.synchronize()
    
    if geo_parameters["extrude"] == "surfaces":
        raise Exception("Extruding surfaces not implemented")

    # Save the model
    gmsh.write(geo_parameters["output_path"]+geo_parameters["output_format"])

    # Finalize gmsh
    gmsh.finalize()

       
def _get_default_parameters():

    default_parameters = {
        "cross_section_path": "input/models/cross_section",
        "cross_section_format": ".geo_unrolled",
        "length": 95.185,
        "number_of_divisions": 31,
        "interpolation": "parabolic",
        "output_path": "input/models/geometry",
        "output_format": ".geo_unrolled",
    }

    return default_parameters

def _check_valid_formats(extension):

    valid_formats = [".geo_unrolled",".msh"]
    valid_formats_str = " ".join(valid_formats)
    if extension in valid_formats:
        return
    else:
        raise Exception(f"[Geometry creation] Output format {extension} not compatible. \
        Valid formats are {valid_formats_str}")


def _check_valid_extrusion(extrusion):

    valid_formats = [".surfaces","lines"]
    valid_formats_str = " ".join(valid_formats)
    if extrusion in valid_formats:
        return
    else:
        raise Exception(f"[Geometry creation] Extrude mode {extrusion} not defined. \
        Valid modes are {valid_formats_str}")