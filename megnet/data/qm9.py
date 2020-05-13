"""
Simple qm9 utils, kept here for historical reasons
"""
from monty.json import MSONable

ATOMNUM2TYPE = {"1": 1, "6": 2, "7": 4, "8": 6, "9": 8}


class AtomNumberToTypeConverter(MSONable):
    """
    Convert atomic number Z into the atomic type in the QM9 dataset
    """
    def __init__(self, mapping=ATOMNUM2TYPE):
        self.mapping = mapping

    def convert(self, l):
        return [self.mapping[str(i)] for i in l]


def ring_to_vector(l):
    """
    Convert the ring sizes vector to a fixed length vector
    For example, l can be [3, 5, 5], meaning that the atom is involved
    in 1 3-sized ring and 2 5-sized ring. This function will convert it into
    [ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0].
    Args:
        l: (list of integer) ring_sizes attributes
    Returns:
        (list of integer) fixed size list with the i-1 th element indicates number of
            i-sized ring this atom is involved in.
    """
    return_l = [0] * 9
    if l:
        for i in l:
            return_l[i - 1] += 1
    return return_l
