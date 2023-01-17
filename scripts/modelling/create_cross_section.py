import gmsh

def create_cross_section(parameters):
    "Creates the cross section of the Nibelungenbrücke from a set of parameters"
    
    # Import parameters
    cs_parameters = _get_default_parameters()
    for key, value in parameters.items():
        cs_parameters[key] = value

    # Sanity checks
    _check_valid_formats(cs_parameters["output_format"])

    #### CREATE THE GEOMETRY ####
    
    #  0. Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    ### 1. CREATE GEOMETRY AT THE PILOT
    
    ## 1.1 Create external geometry

    # Create a new model
    gmsh.model.add("pilot")

    # Define the dimensions of the geometry
    pilot_bottom_height=cs_parameters["pilot_bottom_height"]
    pilot_top_height=cs_parameters["pilot_top_height"]
    pilot_width=cs_parameters["pilot_width"]
    pilot_fly=cs_parameters["pilot_fly"]
    deck_thickness=cs_parameters["deck_thickness"]
    gap_length=cs_parameters["gap_length"]
    wall_thickness=cs_parameters["wall_thickness"]

    # Define the points for the line profile
    # TODO: Find better solution for the tags
    p01 = gmsh.model.geo.addPoint(-gap_length/2 - pilot_width -pilot_fly, 0, 0, tag=1)
    p03 = gmsh.model.geo.addPoint(-gap_length/2 - pilot_width, 0, 0, tag=3)
    p04 = gmsh.model.geo.addPoint(-gap_length/2 - pilot_width, -(pilot_top_height-pilot_bottom_height), 0, tag=4)
    p05 = gmsh.model.geo.addPoint(-gap_length/2 , -(pilot_top_height-pilot_bottom_height), 0, tag=5)
    p06 = gmsh.model.geo.addPoint(-gap_length/2 , 0, 0, tag=6)
    p12 = gmsh.model.geo.addPoint(gap_length/2 + pilot_width +pilot_fly, 0, 0, tag=12)
    p10 = gmsh.model.geo.addPoint(gap_length/2 + pilot_width, 0, 0, tag=10)
    p09 = gmsh.model.geo.addPoint(gap_length/2 + pilot_width, -(pilot_top_height-pilot_bottom_height), 0, tag=9)
    p08 = gmsh.model.geo.addPoint(gap_length/2 , -(pilot_top_height-pilot_bottom_height), 0, tag=8)
    p07 = gmsh.model.geo.addPoint(gap_length/2 , 0, 0, tag=7)

    # Create the lines for the pilot
    l1 = gmsh.model.geo.addLine(p01, p12, tag=1)
    l2 = gmsh.model.geo.addLine(p03, p04, tag=2)
    l3 = gmsh.model.geo.addLine(p04, p05, tag=3)
    l4 = gmsh.model.geo.addLine(p05, p06, tag=4)
    # l5 = gmsh.model.geo.addLine(p06, p07, tag=5)
    l6 = gmsh.model.geo.addLine(p07, p08, tag=6)
    l7 = gmsh.model.geo.addLine(p08, p09, tag=7)
    l8 = gmsh.model.geo.addLine(p09, p10, tag=8)

    # Synchronize the model
    gmsh.model.geo.synchronize()
    
    # Save the model
    gmsh.write(cs_parameters["output_path"]+"_pilot"+cs_parameters["output_format"])

    # Finalize gmsh
    gmsh.finalize()

    ### 2. CREATE GEOMETRY AT THE MIDDLE OF THE SPAN

    #  0. Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Create a new model
    gmsh.model.add("span")

    # Define the dimensions of the geometry
    span_bottom_height=cs_parameters["span_bottom_height"]
    span_top_height=cs_parameters["span_top_height"]
    span_width=cs_parameters["span_width"]
    span_fly=cs_parameters["span_fly"]

    # Define the points for the line profile
    p01 = gmsh.model.geo.addPoint(-gap_length/2 - span_width -span_fly, 0, 0, tag=101)
    p03 = gmsh.model.geo.addPoint(-gap_length/2 - span_width, 0, 0, tag=103)
    p04 = gmsh.model.geo.addPoint(-gap_length/2 - span_width, -(span_top_height-span_bottom_height), 0, tag=104)
    p05 = gmsh.model.geo.addPoint(-gap_length/2 , -(span_top_height-span_bottom_height), 0, tag=105)
    p06 = gmsh.model.geo.addPoint(-gap_length/2 , 0, 0, tag=106)
    p12 = gmsh.model.geo.addPoint(gap_length/2 + span_width +span_fly, 0, 0, tag=112)
    p10 = gmsh.model.geo.addPoint(gap_length/2 + span_width, 0, 0, tag=110)
    p09 = gmsh.model.geo.addPoint(gap_length/2 + span_width, -(span_top_height-span_bottom_height), 0, tag=109)
    p08 = gmsh.model.geo.addPoint(gap_length/2 , -(span_top_height-span_bottom_height), 0, tag=108)
    p07 = gmsh.model.geo.addPoint(gap_length/2 , 0, 0, tag=107)

    # Create the lines for the span
    l1 = gmsh.model.geo.addLine(p01, p12, tag=101)
    l2 = gmsh.model.geo.addLine(p03, p04, tag=102)
    l3 = gmsh.model.geo.addLine(p04, p05, tag=103)
    l4 = gmsh.model.geo.addLine(p05, p06, tag=104)
    # l5 = gmsh.model.geo.addLine(p06, p07, tag=105)
    l6 = gmsh.model.geo.addLine(p07, p08, tag=106)
    l7 = gmsh.model.geo.addLine(p08, p09, tag=107)
    l8 = gmsh.model.geo.addLine(p09, p10, tag=108)

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Save the model
    gmsh.write(cs_parameters["output_path"]+"_span"+cs_parameters["output_format"])

    # Finalize gmsh
    gmsh.finalize()

def _get_default_parameters():

    #             span_top    |---gap_lenth---|    pilot_top
    # p01_____________________________X0_____________________________p12
    # |_______________________________||_______________________________| deck_thickness
    # p02  p03|   |       |   |p06    ||    p07|   |       |   |p10  p11
    #   span_ |   |       |   |       ||       |   |       |   |  pilot_
    #   fly   |   |       |   |       ||       |   |       |   |  fly
    #         |   |       |   |_______||_______|   |       |   |
    #         |   |_______|   |       ||       |   |       |   |
    #      p04|_______________|p05    ||       |   |       |   |wall_thickness
    #              span_width         ||       |   |       |   |
    #             span_bottom         ||       |   |_______|   |
    #                                 ||    p08|_______________|p09
    #                                 ||          pilot_width
    #                                 ||          pilot_bottom
    
    default_parameters = {
        "output_path": "inpu/models/cross_section",
        "output_format": ".geo",
        "pilot_bottom_height": 99.328,
        "pilot_top_height": 105.849,
        "pilot_width": 2.0,
        "pilot_fly": 3.47,
        "span_bottom_height": 102.698,
        "span_top_height": 105.198,
        "span_width": 2.0,
        "span_fly": 3.15,
        "deck_thickness": 0.2,
        "gap_length": 3.70,
        "wall_thickness": 0.4
    }

    return default_parameters

def _check_valid_formats(extension):

    valid_formats = [".geo_unrolled",".msh"]
    valid_formats_str = " ".join(valid_formats)
    if extension in valid_formats:
        return
    else:
        raise Exception(f"[Cross-section creation] Output format {extension} not compatible. \
        Valid formats are {valid_formats_str}")