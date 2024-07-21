'''Collection of helper functions.'''
from __future__ import annotations
from typing import Callable, List

import os
import glob
import logging
from logging.handlers import RotatingFileHandler
import telegram_bot.configuration as config

# Comment ##############################################################
# Code ########################################################################

def start_logger(
        logfile_name: str='llm_detector.log',
        logger_name: str='benchmarking'
) -> Callable:

    '''Sets up logging, returns logger'''

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/{logfile_name}*'):
            os.remove(file)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(config.LOG_LEVEL)

    print(f'Will log to: {config.LOG_PATH}/{logfile_name}')

    handler = RotatingFileHandler(
        f'{config.LOG_PATH}/{logfile_name}',
        encoding = 'utf-8',
        maxBytes = 1 * 1024 * 1024,  # 1 MiB
        backupCount = 5
    )

    formatter = logging.Formatter(config.LOG_PREFIX,
                                  datefmt = '%Y-%m-%d %I:%M:%S %p')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('############################################### ')
    logger.info('########### Starting Telegram Bot ############# ')
    logger.info('############################################### ')

    return logger