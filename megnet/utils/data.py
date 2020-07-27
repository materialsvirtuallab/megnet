"""
Data utitlities
"""
from typing import Tuple

import numpy as np
from pymatgen import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from megnet.config import DataType
from megnet.utils.typing import StructureOrMolecule


def get_graphs_within_cutoff(structure: StructureOrMolecule,
                             cutoff: float = 5.0,
                             numerical_tol: float = 1e-8) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(
            np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array(
            [[1000.0, 0., 0.],
             [0., 1000., 0.],
             [0., 0., 1000.]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
    else:
        raise ValueError('structure type not supported')
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(
        np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = \
        find_points_in_spheres(cart_coords, cart_coords, r=r, pbc=pbc,
                               lattice=lattice_matrix, tol=numerical_tol)
    center_indices = center_indices.astype(DataType.np_int)
    neighbor_indices = neighbor_indices.astype(DataType.np_int)
    images = images.astype(DataType.np_int)
    distances = distances.astype(DataType.np_float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return center_indices[exclude_self], neighbor_indices[exclude_self], \
        images[exclude_self], distances[exclude_self]
