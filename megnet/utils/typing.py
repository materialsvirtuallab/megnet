"""Define several typing for convenient use"""

from typing import Any, Callable, List, Optional, Union

import numpy as np
from pymatgen.core import Molecule, Structure

OptStrOrCallable = Optional[Union[str, Callable[..., Any]]]
StructureOrMolecule = Union[Structure, Molecule]
VectorLike = Union[List[float], np.ndarray]
