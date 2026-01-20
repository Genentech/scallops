import multiprocessing
import os
import threading
from contextlib import ExitStack
from subprocess import check_call

import psutil
from WDL.runtime.task_container import TaskContainer


class LocalRunner(TaskContainer):
    _run_lock = threading.Lock()

    @classmethod
    def global_init(cls, cfg, logger):
        """
        Perform any necessary process-wide initialization of the container backend
        """
        cls._resource_limits = {
            "cpu": multiprocessing.cpu_count(),
            "mem_bytes": psutil.virtual_memory().total,
        }

    @classmethod
    def detect_resource_limits(cls, cfg, logger):
        return cls._resource_limits

    def __init__(self, cfg, run_id, host_dir):
        super().__init__(cfg, run_id, host_dir)
        self.host_dir = host_dir

    def _run(self, logger, terminating, command):
        with ExitStack() as cleanup:
            invocation = self.run_invocation(command)
            cleanup.enter_context(self._run_lock)
            cleanup.enter_context(self.task_running_context())

            return check_call(
                invocation, cwd=os.path.join(self.host_dir, "work"), shell=True
            )

    def run_invocation(self, command):
        return [command]

    def prepare_mounts(self, command):
        return []
