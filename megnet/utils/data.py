from pymatgen.optimization.neighbors import find_points_in_spheres
from pymatgen import Structure, Molecule
from megnet.utils.molecule import MEGNetMolecule
from typing import Union
import numpy as np


def get_graphs_within_cutoff(structure: Union[Structure, MEGNetMolecule, Molecule],
                             cutoff: float = 5.0, numerical_tol: float = 1e-8):
    """
    Get graph representations from structure within cutoff
    Args:
        structure: (pymatgen Structure)
        cutoff: (float) cutoff radius
        numerical_tol: (float) numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
    elif isinstance(structure, MEGNetMolecule) or isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0., 0.], [0., 1000., 0.], [0., 0., 1000.]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
    else:
        raise ValueError('structure type not supported')
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = \
        find_points_in_spheres(cart_coords, cart_coords, r=r, pbc=pbc,
                               lattice=lattice_matrix, tol=numerical_tol)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[exclude_self]
