from pymatgen.analysis.local_env import *
from inspect import getfullargspec


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
        for nn in neighs_dists:
            siw.append({'site': nn,
                        'image': self._get_image(structure, nn),
                        'weight': nn.nn_distance,
                        'site_index': self._get_original_site(structure, nn)})
        return siw

    def get_all_nn_info_old(self, structure):
        nn_info = []
        all_neighbors = structure.get_all_neighbors(self.cutoff, include_index=True,
                                                    include_image=True)
        for n, neighd_dists in enumerate(all_neighbors):
            siw = []
            for _, dist, ind, image in neighd_dists:
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


def serialize(identifier):
    """
    Serialize the local env objects to a dictionary
    Args:
        identifier: (NearNeighbors object/str/None)

    Returns: dictionary or None

    """
    if isinstance(identifier, str):
        return identifier
    elif isinstance(identifier, NearNeighbors):
        args = getfullargspec(identifier.__class__.__init__).args
        d = {"@module": identifier.__class__.__module__,
             "@class": identifier.__class__.__name__}
        for arg in args:
            if arg == 'self':
                continue
            try:
                a = identifier.__getattribute__(arg)
                d[arg] = a
            except AttributeError:
                raise ValueError("Cannot find the argument")
        if hasattr(identifier, "kwargs"):
            d.update(**identifier.kwargs)
        return d
    elif identifier is None:
        return None
    else:
        raise ValueError('Unknown identifier for local environment ', identifier)


def deserialize(config):
    """
    Deserialize the config dict to object
    Args:
        config: (dict) nn_strategy config dict from seralize function

    Returns: object

    """
    if config is None:
        return None
    modname = config['@module']
    classname = config['@class']
    mod = __import__(modname, globals(), locals(), [classname])
    cls_ = getattr(mod, classname)
    data = {k: v for k, v in config.items() if not k.startswith('@')}
    return cls_(**data)


def get(identifier):
    if isinstance(identifier, str):
        return globals()[identifier]
    elif isinstance(identifier, NearNeighbors):
        return identifier
    else:
        raise ValueError('Unknown local environment ', identifier)
