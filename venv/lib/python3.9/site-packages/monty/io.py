"""
Augments Python's suite of IO functions with useful transparent support for
compressed files.
"""

import os
import bz2
import gzip
import lzma
import time
import errno
import mmap
import subprocess
import io
from typing import Union, IO, Generator
from pathlib import Path


def zopen(filename: Union[str, Path], *args, **kwargs) -> IO:
    r"""
    This function wraps around the bz2, gzip, lzma, xz and standard python's open
    function to deal intelligently with bzipped, gzipped or standard text
    files.

    Args:
        filename (str/Path): filename or pathlib.Path.
        *args: Standard args for python open(..). E.g., 'r' for read, 'w' for
            write.
        **kwargs: Standard kwargs for python open(..).

    Returns:
        File-like object. Supports with context.
    """
    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    name, ext = os.path.splitext(filename)
    ext = ext.upper()
    if ext == ".BZ2":
        return bz2.open(filename, *args, **kwargs)
    if ext in (".GZ", ".Z"):
        return gzip.open(filename, *args, **kwargs)
    if ext in (".XZ", ".LZMA"):
        return lzma.open(filename, *args, **kwargs)
    return open(filename, *args, **kwargs)  # pylint: disable=R1732


def reverse_readfile(filename: Union[str, Path]) -> Generator[str, str, None]:
    """
    A much faster reverse read of file by using Python's mmap to generate a
    memory-mapped file. It is slower for very small files than
    reverse_readline, but at least 2x faster for large files (the primary use
    of such a method).

    Args:
        filename (str):
            Name of file to read.

    Yields:
        Lines from the file in reverse order.
    """
    try:
        with zopen(filename, "rb") as f:
            if isinstance(f, (gzip.GzipFile, bz2.BZ2File)):
                for line in reversed(f.readlines()):
                    yield line.decode("utf-8").rstrip()
            else:
                fm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                n = len(fm)
                while n > 0:
                    i = fm.rfind(b"\n", 0, n)
                    yield fm[i + 1 : n].decode("utf-8").strip("\n")
                    n = i
    except ValueError:
        return


def reverse_readline(m_file, blk_size=4096, max_mem=4000000) -> Generator[str, str, None]:
    """
    Generator method to read a file line-by-line, but backwards. This allows
    one to efficiently get data at the end of a file.

    Based on code by Peter Astrand <astrand@cendio.se>, using modifications by
    Raymond Hettinger and Kevin German.
    http://code.activestate.com/recipes/439045-read-a-text-file-backwards
    -yet-another-implementat/

    Reads file forwards and reverses in memory for files smaller than the
    max_mem parameter, or for gzip files where reverse seeks are not supported.

    Files larger than max_mem are dynamically read backwards.

    Args:
        m_file (File): File stream to read (backwards)
        blk_size (int): The buffer size. Defaults to 4096.
        max_mem (int): The maximum amount of memory to involve in this
            operation. This is used to determine when to reverse a file
            in-memory versus seeking portions of a file. For bz2 files,
            this sets the maximum block size.

    Returns:
        Generator that returns lines from the file. Similar behavior to the
        file.readline() method, except the lines are returned from the back
        of the file.
    """
    # Check if the file stream is a bit stream or not
    is_text = isinstance(m_file, io.TextIOWrapper)

    try:
        file_size = os.path.getsize(m_file.name)
    except AttributeError:
        # Bz2 files do not have name attribute. Just set file_size to above
        # max_mem for now.
        file_size = max_mem + 1

    # If the file size is within our desired RAM use, just reverse it in memory
    # GZip files must use this method because there is no way to negative seek
    # For windows, we also read the whole file.
    if file_size < max_mem or isinstance(m_file, gzip.GzipFile) or os.name == "nt":
        for line in reversed(m_file.readlines()):
            yield line.rstrip()
    else:
        if isinstance(m_file, bz2.BZ2File):
            # for bz2 files, seeks are expensive. It is therefore in our best
            # interest to maximize the blk_size within limits of desired RAM
            # use.
            blk_size = min(max_mem, file_size)

        buf = ""
        m_file.seek(0, 2)
        if is_text:
            lastchar = m_file.read(1)
        else:
            lastchar = m_file.read(1).decode("utf-8")

        trailing_newline = lastchar == "\n"

        while 1:
            newline_pos = buf.rfind("\n")
            pos = m_file.tell()
            if newline_pos != -1:
                # Found a newline
                line = buf[newline_pos + 1 :]
                buf = buf[:newline_pos]
                if pos or newline_pos or trailing_newline:
                    line += "\n"
                yield line
            elif pos:
                # Need to fill buffer
                toread = min(blk_size, pos)
                m_file.seek(pos - toread, 0)
                if is_text:
                    buf = m_file.read(toread) + buf
                else:
                    buf = m_file.read(toread).decode("utf-8") + buf
                m_file.seek(pos - toread, 0)
                if pos == toread:
                    buf = "\n" + buf
            else:
                # Start-of-file
                return


class FileLockException(Exception):
    """Exception raised by FileLock."""


class FileLock:
    """
    A file locking mechanism that has context-manager support so you can use
    it in a with statement. This should be relatively cross-compatible as it
    doesn't rely on msvcrt or fcntl for the locking.
    Taken from http://www.evanfosmark.com/2009/01/cross-platform-file-locking
    -support-in-python/
    """

    Error = FileLockException

    def __init__(self, file_name, timeout=10, delay=0.05):
        """
        Prepare the file locker. Specify the file to lock and optionally
        the maximum timeout and the delay between each attempt to lock.

        Args:
            file_name: Name of file to lock.
            timeout: Maximum timeout for locking. Defaults to 10.
            delay: Delay between each attempt to lock. Defaults to 0.05.
        """
        self.file_name = os.path.abspath(file_name)
        self.lockfile = os.path.abspath(file_name) + ".lock"
        self.timeout = float(timeout)
        self.delay = float(delay)
        self.is_locked = False

        if self.delay > self.timeout or self.delay <= 0 or self.timeout <= 0:
            raise ValueError("delay and timeout must be positive with delay " "<= timeout")

    def acquire(self):
        """
        Acquire the lock, if possible. If the lock is in use, it check again
        every `delay` seconds. It does this until it either gets the lock or
        exceeds `timeout` number of seconds, in which case it throws
        an exception.
        """
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except (OSError,) as e:
                if e.errno != errno.EEXIST:
                    raise
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException(f"{self.lockfile}: Timeout occured.")
                time.sleep(self.delay)

        self.is_locked = True

    def release(self):
        """
        Get rid of the lock by deleting the lockfile.
        When working in a `with` statement, this gets automatically
        called at the end.
        """
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def __enter__(self):
        """
        Activated when used in the with statement. Should automatically
        acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type_, value, traceback):
        """
        Activated at the end of the with statement. It automatically releases
        the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()

    def __del__(self):
        """
        Make sure that the FileLock instance doesn't leave a lockfile
        lying around.
        """
        self.release()


def get_open_fds():
    """
    Return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like OS-es.
    """
    pid = os.getpid()
    procs = subprocess.check_output(["lsof", "-w", "-Ff", "-p", str(pid)])
    procs = procs.decode("utf-8")

    return len([s for s in procs.split("\n") if s and s[0] == "f" and s[1:].isdigit()])
