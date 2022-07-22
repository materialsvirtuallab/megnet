"""
Various NearNeighbors strategies to define local environments
of sites in structure/molecule. Most of them are directly
from pymatgen.analysis.local_env. The suitable NearNeighbors
should have get_nn_info method implemented and this method
needs to return a list of dict with each entry having following keys
['site', 'image', 'weight', 'site_index']

the weight will be used as the bond attributes in subsequent graph
construction

"""
from inspect import getfullargspec
from typing import Dict, List, Union

from pymatgen.analysis import local_env
from pymatgen.analysis.local_env import (
    BrunnerNN_real,
    BrunnerNN_reciprocal,
    BrunnerNN_relative,
    CovalentBondNN,
    Critic2NN,
    CrystalNN,
    CutOffDictNN,
    EconNN,
    JmolNN,
    MinimumDistanceNN,
    MinimumOKeeffeNN,
    MinimumVIRENN,
    NearNeighbors,
    OpenBabelNN,
    VoronoiNN,
)
from pymatgen.core import Molecule, Structure


class MinimumDistanceNNAll(NearNeighbors):
    """
    Determine bonded sites by fixed cutoff
    """

    def __init__(self, cutoff: float = 4.0):
        """
        Args:.
            cutoff (float): cutoff radius in Angstrom to look for trial
                near-neighbor sites (default: 4.0).
        """
        self.cutoff = cutoff

    def get_nn_info(self, structure: Structure, n: int) -> List[Dict]:
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
            siw.append(
                {
                    "site": nn,
                    "image": self._get_image(structure, nn),
                    "weight": nn.nn_distance,
                    "site_index": self._get_original_site(structure, nn),
                }
            )
        return siw


class AllAtomPairs(NearNeighbors):
    """
    Get all combinations of atoms as bonds in a molecule
    """

    def get_nn_info(self, molecule: Molecule, n: int) -> List[Dict]:
        """
        Get near neighbor information
        Args:
            molecule (Molecule): pymatgen Molecule
            n (int): number of molecule

        Returns: List of neighbor dictionary

        """
        site = molecule[n]
        siw = []
        for i, s in enumerate(molecule):
            if i != n:
                siw.append({"site": s, "image": None, "weight": site.distance(s), "site_index": i})
        return siw


def serialize(identifier: Union[str, NearNeighbors]):
    """
    Serialize the local env objects to a dictionary
    Args:
        identifier: (NearNeighbors object/str/None)

    Returns: dictionary or None

    """
    if isinstance(identifier, str):
        return identifier
    if isinstance(identifier, NearNeighbors):
        args = getfullargspec(identifier.__class__.__init__).args
        d = {"@module": identifier.__class__.__module__, "@class": identifier.__class__.__name__}
        for arg in args:
            if arg == "self":
                continue
            try:
                a = getattr(identifier, arg)
                d[arg] = a
            except AttributeError:
                raise ValueError("Cannot find the argument")
        if hasattr(identifier, "kwargs"):
            d.update(**identifier.kwargs)
        return d
    if identifier is None:
        return None

    raise ValueError("Unknown identifier for local environment ", identifier)


def deserialize(config: Dict):
    """
    Deserialize the config dict to object
    Args:
        config: (dict) nn_strategy config dict from seralize function

    Returns: object

    """
    if config is None:
        return None
    if ("@module" not in config) or ("@class" not in config):
        raise ValueError("The config dict cannot be loaded")
    modname = config["@module"]
    classname = config["@class"]
    mod = __import__(modname, globals(), locals(), [classname])
    cls_ = getattr(mod, classname)
    data = {k: v for k, v in config.items() if not k.startswith("@")}
    return cls_(**data)


NNDict = {
    i.__name__.lower(): i
    for i in [
        NearNeighbors,
        VoronoiNN,
        JmolNN,
        MinimumDistanceNN,
        OpenBabelNN,
        CovalentBondNN,
        MinimumVIRENN,
        MinimumOKeeffeNN,
        BrunnerNN_reciprocal,
        BrunnerNN_real,
        BrunnerNN_relative,
        EconNN,
        CrystalNN,
        CutOffDictNN,
        Critic2NN,
        MinimumDistanceNNAll,
        AllAtomPairs,
    ]
}


def get(identifier: Union[str, Dict, NearNeighbors]) -> NearNeighbors:
    """
    Deserialize the NearNeighbors
    Args:
        identifier (str, dict or NearNeighbors): target for deserialize

    Returns: NearNeighbors instance

    """
    # deserialize NearNeighbor from str
    if isinstance(identifier, str):
        if identifier.lower() in NNDict:
            return NNDict.get(identifier.lower())
        # try pymatgen's local_env module
        nn = getattr(local_env, identifier, None)
        if nn is not None:
            return nn

    if isinstance(identifier, dict):
        return deserialize(identifier)

    if isinstance(identifier, NearNeighbors):
        return identifier

    raise ValueError(f"{identifier} not identified")
