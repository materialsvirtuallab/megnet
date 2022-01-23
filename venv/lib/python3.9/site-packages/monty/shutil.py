"""
Copying and zipping utilities. Works on directories mostly.
"""

import os
import shutil
import warnings
from gzip import GzipFile

from .io import zopen


def copy_r(src, dst):
    """
    Implements a recursive copy function similar to Unix's "cp -r" command.
    Surprisingly, python does not have a real equivalent. shutil.copytree
    only works if the destination directory is not present.

    Args:
        src (str): Source folder to copy.
        dst (str): Destination folder.
    """

    abssrc = os.path.abspath(src)
    absdst = os.path.abspath(dst)
    try:
        os.makedirs(absdst)
    except OSError:
        # If absdst exists, an OSError is raised. We ignore this error.
        pass
    for f in os.listdir(abssrc):
        fpath = os.path.join(abssrc, f)
        if os.path.isfile(fpath):
            shutil.copy(fpath, absdst)
        elif not absdst.startswith(fpath):
            copy_r(fpath, os.path.join(absdst, f))
        else:
            warnings.warn(f"Cannot copy {fpath} to itself")


def gzip_dir(path, compresslevel=6):
    """
    Gzips all files in a directory. Note that this is different from
    shutil.make_archive, which creates a tar archive. The aim of this method
    is to create gzipped files that can still be read using common Unix-style
    commands like zless or zcat.

    Args:
        path (str): Path to directory.
        compresslevel (int): Level of compression, 1-9. 9 is default for
            GzipFile, 6 is default for gzip.
    """
    for root, _, files in os.walk(path):
        for f in files:
            full_f = os.path.abspath(os.path.join(root, f))
            if not f.lower().endswith("gz") and not os.path.isdir(full_f):
                with open(full_f, "rb") as f_in, GzipFile(f"{full_f}.gz", "wb", compresslevel=compresslevel) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                shutil.copystat(full_f, f"{full_f}.gz")
                os.remove(full_f)


def compress_file(filepath, compression="gz"):
    """
    Compresses a file with the correct extension. Functions like standard
    Unix command line gzip and bzip2 in the sense that the original
    uncompressed files are not retained.

    Args:
        filepath (str): Path to file.
        compression (str): A compression mode. Valid options are "gz" or
            "bz2". Defaults to "gz".
    """
    if compression not in ["gz", "bz2"]:
        raise ValueError("Supported compression formats are 'gz' and 'bz2'.")
    if not filepath.lower().endswith(f".{compression}"):
        with open(filepath, "rb") as f_in, zopen(f"{filepath}.{compression}", "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(filepath)


def compress_dir(path, compression="gz"):
    """
    Recursively compresses all files in a directory. Note that this
    compresses all files singly, i.e., it does not create a tar archive. For
    that, just use Python tarfile class.

    Args:
        path (str): Path to parent directory.
        compression (str): A compression mode. Valid options are "gz" or
            "bz2". Defaults to gz.
    """
    for parent, subdirs, files in os.walk(path):
        for f in files:
            compress_file(os.path.join(parent, f), compression=compression)


def decompress_file(filepath):
    """
    Decompresses a file with the correct extension. Automatically detects
    gz, bz2 or z extension.

    Args:
        filepath (str): Path to file.
        compression (str): A compression mode. Valid options are "gz" or
            "bz2". Defaults to "gz".
    """
    toks = filepath.split(".")
    file_ext = toks[-1].upper()
    if file_ext in ["BZ2", "GZ", "Z"]:
        with open(".".join(toks[0:-1]), "wb") as f_out, zopen(filepath, "rb") as f_in:
            f_out.writelines(f_in)
        os.remove(filepath)


def decompress_dir(path):
    """
    Recursively decompresses all files in a directory.

    Args:
        path (str): Path to parent directory.
    """
    for parent, subdirs, files in os.walk(path):
        for f in files:
            decompress_file(os.path.join(parent, f))


def remove(path, follow_symlink=False):
    """
    Implements an remove function that will delete files, folder trees and
    symlink trees

    1.) Remove a file
    2.) Remove a symlink and follow into with a recursive rm if follow_symlink
    3.) Remove directory with rmtree

    Args:
        path (str): path to remove
        follow_symlink(bool): follow symlinks and removes whatever is in them
    """
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.islink(path):
        if follow_symlink:
            remove(os.readlink(path))
        os.unlink(path)
    else:
        shutil.rmtree(path)
