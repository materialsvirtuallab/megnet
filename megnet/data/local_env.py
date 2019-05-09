from pymatgen.analysis.local_env import *


class MinimumDistanceNNAll(MinimumDistanceNN):
    """
    Determine bonded sites by fixed cutoff

    Args:.
        cutoff (float): cutoff radius in Angstrom to look for trial
            near-neighbor sites (default: 4.0).
    """

    def __init__(self, cutoff=4.0):
        self.cutoff = cutoff

    def get_nn_info(self, structure, n):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the closest neighbor
        distance-based method.

        Args:
            structure (Structure): input structure.
            n (integer): index of site for which to determine near
                neighbors.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a neighbor site, its image location,
                and its weight.
        """

        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)

        siw = []
        for s, dist in neighs_dists:
            w = dist
            siw.append({'site': s,
                        'image': self._get_image(structure, s),
                        'weight': w,
                        'site_index': self._get_original_site(structure, s)})
        return siw

    def get_all_nn_info(self, structure):
        nn_info = []
        all_neighbors = structure.get_all_neighbors(self.cutoff, include_index=True, include_image=True, include_site=False)
        for n, neighd_dists in enumerate(all_neighbors):
            siw = []
            for dist, ind, image in neighd_dists:
                siw.append({'image': image,
                            'weight': dist,
                            'site_index': ind})
            nn_info.append(siw)
        return nn_info


class AllAtomPairs(NearNeighbors):
    """
    Get all combinations of atoms as bonds in a molecule
    """
    def get_nn_info(self, molecule, n):
        site = molecule[n]
        siw = []
        for i, s in enumerate(molecule):
            if i != n:
                siw.append({'site': s,
                            'image': None,
                            'weight': site.distance(s),
                            'site_index': i})
        return siw


def get(identifier):
    if isinstance(identifier, str):
        return globals()[identifier]
    elif isinstance(identifier, NearNeighbors):
        return identifier
    else:
        raise ValueError('Unknown local environment ', identifier)

