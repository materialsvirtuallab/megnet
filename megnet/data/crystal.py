"""
Crystal graph related
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Element, Structure

from megnet.data.graph import Converter, StructureGraph, StructureGraphFixedRadius

MODULE_DIR = Path(__file__).parent.absolute()


def get_elemental_embeddings() -> dict:
    """
    Provides the pre-trained elemental embeddings using formation energies,
    which can be used to speed up the training of other models. The embeddings
    are also extremely useful elemental descriptors that encode chemical
    similarity that may be used in other ways. See

    "Graph Networks as a Universal Machine Learning Framework for Molecules
    and Crystals", https://arxiv.org/abs/1812.05055

    :return: dict of elemental embeddings as {symbol: length 16 string}
    """
    return loadfn(MODULE_DIR / "resources" / "elemental_embedding_1MEGNet_layer.json")


class CrystalGraph(StructureGraphFixedRadius):
    """
    Convert a crystal into a graph with z as atomic feature and distance as bond feature
    one can optionally include state features
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "MinimumDistanceNNAll",
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
        cutoff: float = 5.0,
    ):
        """
        Convert the structure into crystal graph
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff
        )


class CrystalGraphWithBondTypes(StructureGraph):
    """
    Overwrite the bond attributes with bond types, defined simply by
    the metallicity of the atoms forming the bond. Three types of
    scenario is considered, nonmetal-nonmetal (type 0), metal-nonmetal (type 1), and
    metal-metal (type 2)

    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "VoronoiNN",
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
    ):
        """

        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
        """
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter)

    def convert(self, structure: Structure, state_attributes: list | None = None) -> dict:
        """
        Convert structure into graph
        Args:
            structure (Structure): pymatgen Structure
            state_attributes (list): state attributes

        Returns: graph dictionary

        """
        graph = super().convert(structure, state_attributes=state_attributes)
        return self._get_bond_type(graph)

    @staticmethod
    def _get_bond_type(graph) -> dict:
        new_graph = deepcopy(graph)
        elements = [Element.from_Z(i) for i in graph["atom"]]
        for k, (i, j) in enumerate(zip(graph["index1"], graph["index2"])):
            new_graph["bond"][k] = elements[i].is_metal + elements[j].is_metal
        return new_graph


class _AtomEmbeddingMap(Converter):
    """
    Fixed Atom embedding map, used with CrystalGraphDisordered
    """

    def __init__(self, embedding_dict: dict | None = None):
        """
        Args:
            embedding_dict (dict): element to element vector dictionary
        """
        if embedding_dict is None:
            embedding_dict = get_elemental_embeddings()
        self.embedding_dict = embedding_dict

    def convert(self, atoms: list) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                emb += np.array(self.embedding_dict[k]) * v
            features.append(emb)
        return np.array(features).reshape((len(atoms), -1))


class CrystalGraphDisordered(StructureGraphFixedRadius):
    """
    Enable disordered site predictions
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "MinimumDistanceNNAll",
        atom_converter: Converter = _AtomEmbeddingMap(),
        bond_converter: Converter | None = None,
        cutoff: float = 5.0,
    ):
        """
        Convert the structure into crystal graph
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff
        )

    @staticmethod
    def get_atom_features(structure) -> list[dict]:
        """
        For a structure return the list of dictionary for the site occupancy
        for example, Fe0.5Ni0.5 site will be returned as {"Fe": 0.5, "Ni": 0.5}

        Args:
            structure (Structure): pymatgen Structure with potential site disorder

        Returns:
            a list of site fraction description
        """
        return [i.species.as_dict() for i in structure.sites]
