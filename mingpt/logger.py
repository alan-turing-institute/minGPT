import logging

logging.basicConfig(
    datefmt=r"%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)8s] %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
