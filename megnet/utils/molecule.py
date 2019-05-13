from pymatgen import Molecule
import numpy as np


class MEGNetMolecule(Molecule):
    def get_all_neighbors(self, cutoff, include_index=True, include_site=False, include_image=True, **kwargs):
        """

        Args:
            cutoff: float, cutoff radius
            include_index: bool, whether to include the site index
            include_site: bool, whether to include site
            include_image: bool, whether to include dummy image

        Returns:
            list of list, neighbors for each site
        """

        dist = self.distance_matrix
        neighbors = []
        for i in dist:
            inds = np.array(np.where(i <= cutoff)[0], dtype='int')
            d = i[i <= cutoff]
            images = [0] * len(d)
            neighbor = []
            for k, l, m in zip(d, inds, images):
                entry = (k, )
                if include_index:
                    entry += (l, )
                if include_image:
                    entry += (m, )
                if include_site:
                    entry += (self[l], )
                neighbor.append(entry)
            neighbors.append(neighbor)
        return neighbors
