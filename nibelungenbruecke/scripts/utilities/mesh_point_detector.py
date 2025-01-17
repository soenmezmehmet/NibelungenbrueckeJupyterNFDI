import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_closest_entity, create_midpoint_tree


def is_point_inside_cell_hardcoded(cell_geometry, point):
    """
    Check if a point is inside a specific cell using hardcoded geometric calculations.

    Args:
        cell_geometry (np.ndarray): Coordinates of the cell's vertices (e.g., 3x3 for a triangle or 4x3 for a tetrahedron).
        point (array-like): The coordinates of the point to check.

    Returns:
        bool: True if the point is inside the cell, False otherwise.
    """
    point = np.array(point, dtype=np.float64)

    # For 2D (triangle) or 3D (tetrahedron)
    num_vertices, dim = cell_geometry.shape

    if dim == 2 and num_vertices == 3:  # 2D Triangle
        # Compute barycentric coordinates
        v0 = cell_geometry[1] - cell_geometry[0]
        v1 = cell_geometry[2] - cell_geometry[0]
        v2 = point - cell_geometry[0]

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute the denominator of barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return False  # Degenerate triangle

        # Compute barycentric coordinates
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        # Check if point is inside the triangle
        return u >= 0 and v >= 0 and (u + v) <= 1

    elif dim == 3 and num_vertices == 4:  # 3D Tetrahedron
        # Define vectors for the edges of the tetrahedron
        v0 = cell_geometry[1] - cell_geometry[0]
        v1 = cell_geometry[2] - cell_geometry[0]
        v2 = cell_geometry[3] - cell_geometry[0]
        v = point - cell_geometry[0]

        # Compute determinants
        d0 = np.linalg.det([v0, v1, v2])
        if d0 == 0:
            return False  # Degenerate tetrahedron

        # Compute barycentric coordinates
        d1 = np.linalg.det([v, v1, v2]) / d0
        d2 = np.linalg.det([v0, v, v2]) / d0
        d3 = np.linalg.det([v0, v1, v]) / d0

        # Check if point is inside the tetrahedron
        return d1 >= 0 and d2 >= 0 and d3 >= 0 and (d1 + d2 + d3) <= 1

    else:
        raise ValueError("Unsupported cell geometry. Only 2D triangles and 3D tetrahedra are supported.")

def is_point_inside_mesh(point, mesh):
    """
    Check if a point is inside the mesh and return the containing cell index.

    Args:
        point (array-like): The coordinates of the point to check.
        mesh (dolfinx.mesh.Mesh): The mesh object.

    Returns:
        (bool, int): A tuple where the first element indicates if the point is inside,
                     and the second element is the cell index if inside, or -1 otherwise.
    """
    # Create a bounding box tree for the mesh
    tree = BoundingBoxTree(mesh, mesh.topology.dim)

    # Use compute_collisions to find candidate cells
    points = np.array([point], dtype=np.float64)
    colliding_cells = compute_collisions(tree, points)

    for cell_index in colliding_cells.links(0):
        # Get the coordinates of the cell's vertices
        cell_geometry = mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(cell_index)]
        if is_point_inside_cell_hardcoded(cell_geometry, point):
            return True, cell_index

    return False, -1

def project_point_onto_mesh(point, mesh):
    """
    Project a point onto the closest point on the mesh (not limited to nodes).

    Args:
        point (array-like): Coordinates of the point [x, y, z].
        mesh (dolfinx.mesh.Mesh): The mesh object.

    Returns:
        np.ndarray: Coordinates of the closest point on the mesh.
    """
    # Create bounding box trees for facets and midpoints
    facet_tree = BoundingBoxTree(mesh, mesh.topology.dim - 1)
    midpoint_tree = create_midpoint_tree(
        mesh, mesh.topology.dim - 1,
        np.arange(mesh.topology.index_map(mesh.topology.dim - 1).size_local, dtype=np.int32)
    )

    # Compute the closest entity to the point
    points = np.array([point], dtype=np.float64)
    closest_entity_index = compute_closest_entity(facet_tree, midpoint_tree, mesh, points)[0]

    if closest_entity_index == -1:
        raise RuntimeError("Failed to find the closest entity to the point.")

    # Retrieve the vertices of the closest facet
    facet_connectivity = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    facet_vertices = mesh.geometry.x[facet_connectivity.links(closest_entity_index)]

    # Compute the projection using a centroid approximation
    centroid = np.mean(facet_vertices, axis=0)
    return centroid


def query_point(point, mesh):
    """
    Query if a point is inside the mesh, and if not, project it onto the closest point.

    Args:
        point (array-like): Coordinates of the point [x, y, z].
        mesh (dolfinx.mesh.Mesh): The mesh object.

    Returns:
        (np.ndarray, bool): Closest point on the mesh and whether the point is inside the mesh.
    """
    #path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/models/mesh.msh'
    #mesh, cell_tags, facet_tags = gmshio.read_from_msh(path, MPI.COMM_WORLD, 0)
    
    inside, cell_index = is_point_inside_mesh(point, mesh)
    if inside:
        return np.array(point), True

    closest_point = project_point_onto_mesh(point, mesh)
    return closest_point, False


if __name__ == "__main__":

    # Test point
    point = [0.83539223, -0.06666667, 0.36112456]
    result_point, is_inside = query_point(point)
    print(f"Point {point} is inside the mesh: {is_inside}")
    print(f"Closest point on the mesh: {result_point}")
    
    point = [0.5, -0.1, 0.5]
    result_point, is_inside = query_point(point)
    print(f"Point {point} is inside the mesh: {is_inside}")
    print(f"Closest point on the mesh: {result_point}")
    
    
    point = [110.5, -10.1, 10.5]
    result_point, is_inside = query_point(point)
    print(f"Point {point} is inside the mesh: {is_inside}")
    print(f"Closest point on the mesh: {result_point}")


    point = [80, 0.0, 0.0]
    result_point, is_inside = query_point(point)
    print(f"Point {point} is inside the mesh: {is_inside}")
    print(f"Closest point on the mesh: {result_point}")