"""
Path based methods, e.g., which, zpath, etc.
"""
import os

from monty.fnmatch import WildCard
from monty.string import list_strings


def which(cmd):
    """
    Returns full path to a executable.

    Args:
        cmd (str): Executable command to search for.

    Returns:
        (str) Full path to command. None if it is not found.

    Example::

        full_path_to_python = which("python")
    """

    def is_exe(fp):
        return os.path.isfile(fp) and os.access(fp, os.X_OK)

    fpath, fname = os.path.split(cmd)
    if fpath:
        if is_exe(cmd):
            return cmd
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, cmd)
            if is_exe(exe_file):
                return exe_file
    return None


def zpath(filename):
    """
    Returns an existing (zipped or unzipped) file path given the unzipped
    version. If no path exists, returns the filename unmodified.

    Args:
        filename: filename without zip extension

    Returns:
        filename with a zip extension (unless an unzipped version
        exists). If filename is not found, the same filename is returned
        unchanged.
    """
    for ext in ["", ".gz", ".GZ", ".bz2", ".BZ2", ".z", ".Z"]:
        zfilename = f"{filename}{ext}"
        if os.path.exists(zfilename):
            return zfilename
    return filename


def find_exts(top, exts, exclude_dirs=None, include_dirs=None, match_mode="basename"):
    """
    Find all files with the extension listed in `exts` that are located within
    the directory tree rooted at `top` (including top itself, but excluding
    '.' and '..')

    Args:
        top (str): Root directory
        exts (str or list of strings): List of extensions.
        exclude_dirs (str): Wildcards used to exclude particular directories.
            Can be concatenated via `|`
        include_dirs (str): Wildcards used to select particular directories.
            `include_dirs` and `exclude_dirs` are mutually exclusive
        match_mode (str): "basename" if  match should be done on the basename.
            "abspath" for absolute path.

    Returns:
        (list of str): Absolute paths of the files.

    Examples::

        # Find all pdf and ps files starting from the current directory.
        find_exts(".", ("pdf", "ps"))

        # Find all pdf files, exclude hidden directories and dirs whose name
        # starts with `_`
        find_exts(".", "pdf", exclude_dirs="_*|.*")

        # Find all ps files, in the directories whose basename starts with
        # output.
        find_exts(".", "ps", include_dirs="output*"))
    """
    exts = list_strings(exts)

    # Handle file!
    if os.path.isfile(top):
        return [os.path.abspath(top)] if any(top.endswith(ext) for ext in exts) else []

    # Build shell-style wildcards.
    if exclude_dirs is not None:
        exclude_dirs = WildCard(exclude_dirs)

    if include_dirs is not None:
        include_dirs = WildCard(include_dirs)

    mangle = dict(basename=os.path.basename, abspath=os.path.abspath)[match_mode]

    # Assume directory
    paths = []
    for dirpath, dirnames, filenames in os.walk(top):
        dirpath = os.path.abspath(dirpath)

        if exclude_dirs and exclude_dirs.match(mangle(dirpath)):
            continue
        if include_dirs and not include_dirs.match(mangle(dirpath)):
            continue

        for filename in filenames:
            if any(filename.endswith(ext) for ext in exts):
                paths.append(os.path.join(dirpath, filename))

    return paths
