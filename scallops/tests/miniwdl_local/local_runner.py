import logging
import os
import shlex
import subprocess
from collections.abc import Callable
from contextlib import ExitStack

from WDL._util import StructuredLogMessage as _
from WDL._util import TailLogger
from WDL.runtime.backend.cli_subprocess import (
    SubprocessBase,
    _SubprocessScheduler,
    config,
)
from WDL.runtime.error import Terminated


class LocalRunner(SubprocessBase):
    @property
    def cli_name(self) -> str:
        return "cli"

    def _run_invocation(
        self, logger: logging.Logger, cleanup: ExitStack, image: str
    ) -> list[str]:
        return []

    @classmethod
    def global_init(cls, cfg: config.Loader, logger: logging.Logger) -> None:
        pass

    def _pull(self, logger: logging.Logger, cleanup: ExitStack) -> str:
        return ""

    def prepare_mounts(self) -> list[tuple[str, str, bool]]:
        return []

    # need to override cwd to cwd=os.path.join(self.host_dir, "work"))
    def _run(
        self, logger: logging.Logger, terminating: Callable[[], bool], command: str
    ) -> int:
        with ExitStack() as cleanup:
            # await cpu & memory availability
            cpu_reservation = self.runtime_values.get("cpu", 0)
            memory_reservation = self.runtime_values.get("memory_reservation", 0)
            scheduler = _SubprocessScheduler(cpu_reservation, memory_reservation)
            cleanup.enter_context(scheduler)
            logger.info(
                _(
                    "provisioned",
                    seconds_waited=scheduler.delay,
                    cpu=cpu_reservation,
                    mem_bytes=memory_reservation,
                )
            )

            # pull image if needed
            image = self._pull(logger, cleanup)

            # prepare loggers
            cli_log_filename = self.host_cli_log_txt(self.cli_name)
            cli_log = cleanup.enter_context(open(cli_log_filename, "wb"))
            cli_logger = logger.getChild(self.cli_name)
            poll_stderr = cleanup.enter_context(self.poll_stderr_context(logger))
            poll_cli_log = cleanup.enter_context(
                TailLogger(
                    logger,
                    cli_log_filename,
                    lambda msg: cli_logger.info(msg.rstrip()),
                    level=logging.INFO,
                )
            )

            # prepare command & environment
            # we set the environment variables at the beginning of the command script because:
            # 1) --env is subject to command line length limitations
            # 2) --env-file isn't implemented consistently wrt quoting, escaping, etc.
            with open(os.path.join(self.host_dir, "command"), "w") as outfile:
                outfile.writelines(
                    f"export {k}={shlex.quote(v)}\n"
                    for k, v in self.runtime_values.get("env", {}).items()
                )
                outfile.write(command)

            # start subprocess
            invocation = self._run_invocation(logger, cleanup, image) + [
                "/bin/sh",
                "-c",
                self.cfg.get("task_runtime", "command_shell")
                + " ../command >> ../stdout.txt 2>> ../stderr.txt",
            ]
            proc = subprocess.Popen(
                invocation,
                stdout=cli_log,
                stderr=subprocess.STDOUT,
                cwd=os.path.join(self.host_dir, "work"),
            )

            #  logger.notice(_(f"{self.cli_name} run", pid=proc.pid, log=cli_log_filename))
            cleanup.enter_context(self.task_running_context())

            # long-poll for completion
            exit_code = None
            while exit_code is None:
                if terminating():
                    proc.terminate()
                try:
                    exit_code = proc.wait(1)
                except subprocess.TimeoutExpired:
                    pass
                poll_stderr()
                cli_log.flush()
                poll_cli_log()
            if terminating():
                raise Terminated()
        assert isinstance(exit_code, int)
        return exit_code
