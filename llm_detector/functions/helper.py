'''Collection of helper functions.'''

import os
import glob
import logging
from logging.handlers import RotatingFileHandler
import llm_detector.configuration as config


def start_logger():
    '''Sets up logging, returns logger'''

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/*.log*'):
            os.remove(file)

    # Create logger
    logger = logging.getLogger('llm_detector')
    logger.setLevel(config.LOG_LEVEL)

    handler = RotatingFileHandler(
        f'{config.LOG_PATH}/llm_detector.log',
        encoding='utf-8',
        maxBytes=32 * 1024 * 1024,  # 32 MiB,
        backupCount=5
    )

    formatter = logging.Formatter(config.LOG_PREFIX, datefmt='%Y-%m-%d %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('############################################### ')
    logger.info('########### Starting LLM detector ############# ')
    logger.info('############################################### ')

    return logger
