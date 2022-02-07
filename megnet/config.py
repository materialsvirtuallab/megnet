"""Data types"""
import numpy as np
import tensorflow as tf

DTYPES = {
    "float32": {"numpy": np.float32, "tf": tf.float32},
    "float16": {"numpy": np.float16, "tf": tf.float16},
    "int32": {"numpy": np.int32, "tf": tf.int32},
    "int16": {"numpy": np.int16, "tf": tf.int16},
}


class DataType:
    """
    Data types for tensorflow. This enables users to choose
    from 32-bit float and int, and 16-bit float and int
    """

    np_float = np.float32
    np_int = np.int32
    tf_float = tf.float32
    tf_int = tf.int32

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]
        cls.tf_float = DTYPES[float_key]["tf"]
        cls.np_int = DTYPES[int_key]["numpy"]
        cls.tf_int = DTYPES[int_key]["tf"]


def set_global_dtypes(data_type) -> None:
    """
    Function to set the data types
    Args:
        data_type (str): '16' or '32'
    Returns:

    """
    DataType.set_dtype(data_type)
