import getpass
import os

import ray

from syftr.configuration import cfg
from syftr.logger import logger


def ray_init(force_remote: bool = False):
    if ray.is_initialized():
        logger.warning(
            "Using existing ray client with address '%s'", ray.client().address
        )
    else:
        address = cfg.ray.remote_endpoint if force_remote else None

        if address is None:
            username = getpass.getuser()
            # Use MATHEW_HOME for Ray temp directory to avoid /tmp disk space issues
            # Keep path short to avoid Unix socket path length limit (107 bytes)
            mathew_home = os.environ.get("MATHEW_HOME", "/tmp")
            ray_tmpdir = os.path.join(mathew_home, "ray")
            os.makedirs(ray_tmpdir, exist_ok=True)
            logger.info(
                "Using local ray client with temporary directory '%s'", ray_tmpdir
            )
            os.environ["RAY_TMPDIR"] = ray_tmpdir

        ray.init(
            address=address,
            logging_level=cfg.logging.level,
        )
