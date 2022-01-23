"""
Useful additional functions for operators
"""
import operator


def operator_from_str(op):
    """
    Return the operator associated to the given string `op`.

    raises:
        `KeyError` if invalid string.
    """
    d = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "%": operator.mod,
        "^": operator.xor,
    }

    try:
        d["/"] = operator.truediv
    except AttributeError:
        pass

    return d[op]
