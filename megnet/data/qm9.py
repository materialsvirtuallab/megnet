"""
Simple qm9 utils, kept here for historical reasons
"""
from monty.json import MSONable


ATOMNUM2TYPE = {"1": 1, "6": 2, "7": 4, "8": 6, "9": 8}


class AtomNumberToTypeConverter(MSONable):
    """
    Convert atomic number Z into the atomic type in the QM9 dataset.
    This is specifically used for this problem, do not use it elsewhere.
    The code is here for historical reasons.
    """
    def __init__(self, mapping=ATOMNUM2TYPE):
        """
        Atomic number to atomic type converter
        Args:
            mapping (dict): mapping dictionary
        """
        self.mapping = mapping

    def convert(self, z_list: list) -> list:
        """
        Convert the atomic number list to atomic type list
        Args:
            z_list (list of integer): atomic number list

        Returns: list of integer, atomic type list

        """
        return [self.mapping[str(i)] for i in z_list]


def ring_to_vector(z_list: list, max_size: int = 9) -> list:
    """
    Convert the ring sizes vector to a fixed length vector
    For example, l can be [3, 5, 5], meaning that the atom is involved
    in 1 3-sized ring and 2 5-sized ring. This function will convert it into
    [ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0].
    Args:
        z_list: (list of integer) ring_sizes attributes
        max_size: (int) maximum number of atoms in the ring
    Returns:
        (list of integer) fixed size list with the i-1 th element indicates number of
            i-sized ring this atom is involved in.
    """
    return_l = [0] * max_size
    if z_list:
        for i in z_list:
            return_l[i - 1] += 1
    return return_l
