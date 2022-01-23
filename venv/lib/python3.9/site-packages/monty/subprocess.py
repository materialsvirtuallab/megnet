"""
Calling shell processes.
"""
import shlex
import threading
import traceback
from subprocess import Popen, PIPE

from .string import is_string


__author__ = "Matteo Giantomass"
__copyright__ = "Copyright 2014, The Materials Virtual Lab"
__version__ = "0.1"
__maintainer__ = "Matteo Giantomassi"
__email__ = "gmatteo@gmail.com"
__date__ = "10/26/14"


class Command:
    """
    Enables to run subprocess commands in a different thread with TIMEOUT
    option.

    Based on jcollado's solution:
        http://stackoverflow.com/questions/1191374/subprocess-with-timeout/4825933#4825933
    and
        https://gist.github.com/kirpit/1306188

    .. attribute:: retcode

        Return code of the subprocess

    .. attribute:: killed

        True if subprocess has been killed due to the timeout

    .. attribute:: output

        stdout of the subprocess

    .. attribute:: error

        stderr of the subprocess

    Example:
        com = Command("sleep 1").run(timeout=2)
        print(com.retcode, com.killed, com.output, com.output)
    """

    def __init__(self, command):
        """
        :param command: Command to execute
        """
        if is_string(command):
            command = shlex.split(command)
        self.command = command
        self.process = None
        self.retcode = None
        self.output, self.error = "", ""
        self.killed = False

    def __str__(self):
        return f"command: {self.command}, retcode: {self.retcode}"

    def run(self, timeout=None, **kwargs):
        """
        Run a command in a separated thread and wait timeout seconds.
        kwargs are keyword arguments passed to Popen.

        Return: self
        """

        def target(**kw):
            try:
                # print('Thread started')
                with Popen(self.command, **kw) as self.process:
                    self.output, self.error = self.process.communicate()
                    self.retcode = self.process.returncode
                # print('Thread stopped')
            except Exception:
                self.error = traceback.format_exc()
                self.retcode = -1

        # default stdout and stderr
        if "stdout" not in kwargs:
            kwargs["stdout"] = PIPE

        if "stderr" not in kwargs:
            kwargs["stderr"] = PIPE

        # thread
        thread = threading.Thread(target=target, kwargs=kwargs)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # print("Terminating process")
            self.process.terminate()
            self.killed = True
            thread.join()

        return self
