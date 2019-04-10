from pymatgen.analysis.local_env import MinimumDistanceNN

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
