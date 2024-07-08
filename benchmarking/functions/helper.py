'''Collection of helper functions.'''
from __future__ import annotations
from typing import Callable

import os
import glob
import argparse
import logging
from logging.handlers import RotatingFileHandler
import benchmarking.configuration as config

# Comment ##############################################################
# Code ########################################################################

def parser_formatter(width: int=76) -> Callable:
    '''Makes lambda function to format argparse help'''

    return lambda prog: argparse.HelpFormatter(prog, max_help_position=width)

def parse_args() -> dict:
    '''Make argparse parser, set arguments, get and parse values
    from user, returns args dict.'''

    parser = argparse.ArgumentParser(
        prog = 'main.py',
        description = ('Launcher for project. Select task to run via '
                       'command line arguments:'),
        epilog = 'Bottom text',
        formatter_class = parser_formatter()
    )

    # Add arguments
    parser.add_argument(
        '--binoculars',
        required = False,
        choices = ['True', 'False'],
        default = 'False',
        help = 'Wether or not to run binoculars',
        metavar = '<BOOL>'
    )

    parser.add_argument(
        '--run-benchmark',
        action = 'append',
        nargs = 2,
        help = ('Run benchmark by specifying path to experiment configuration '
              'file and True/False to resume old run'),
        metavar = ('<CONFIG FILE PATH>', '<RESUME BOOL>')
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def start_logger(logfile_name: str='llm_detector.log') -> Callable:
    '''Sets up logging, returns logger'''

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/{logfile_name}*'):
            os.remove(file)

    # Create logger
    logger = logging.getLogger('llm_detector')
    logger.setLevel(config.LOG_LEVEL)

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
    logger.info('########### Starting LLM detector ############# ')
    logger.info('############################################### ')

    return logger
