import os
import sys

from loguru import logger

file_logging_level = os.getenv("AUTOEMBEDS_FILE_LOGGING_LEVEL", "INFO")
std_out_logging_level = os.getenv("AUTOEMBEDS_STDOUT_LOGGING_LEVEL", "INFO")

logger.remove()
logger.add("auto_embeds.log", rotation="10 MB", level=file_logging_level)
logger.add(sys.stdout, level=std_out_logging_level)
