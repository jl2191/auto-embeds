import os

from loguru import logger

log_level = os.getenv("LOGURU_LEVEL", "INFO")
logger.add("auto_embeds.log", rotation="10 MB", level=log_level)
