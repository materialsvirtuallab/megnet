from pymatgen.optimization.neighbors import find_points_in_spheres
from pymatgen import Structure
import numpy as np


def get_graphs_within_cutoff(structure: Structure, cutoff: float = 5.0, numerical_tol: float = 1e-8):
    """
    Get graph representations from structure within cutoff
    Args:
        structure: (pymatgen Structure)
        cutoff: (float) cutoff radius
        numerical_tol: (float) numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
    r = float(cutoff)
    center_indices, neighbor_indices, images, distances = \
        find_points_in_spheres(cart_coords, cart_coords, r=r,
                               pbc=np.array([1, 1, 1], dtype=int),
                               lattice=lattice_matrix, tol=numerical_tol)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)

    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[exclude_self]
