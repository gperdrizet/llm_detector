'''Collection of helper functions.'''

import os
import glob
import argparse
import logging
from logging.handlers import RotatingFileHandler
import llm_detector_benchmarking.configuration as config

def parse_args() -> dict:
    '''Make argparse parser, set arguments, get and parse values
    from user, returns args dict.'''

    # Instantiate command line argument parser
    parser = argparse.ArgumentParser(
        prog = 'main.py',
        description = 'Launcher for project. Select task to run via command line arguments:',
        epilog = 'Bottom text',
        formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=80)
    )

    # Add arguments
    parser.add_argument(
        '--binoculars',
        required=False,
        choices=['True', 'False'],
        default='False',
        help='Wether or not to run binoculars',
        metavar='BOOL'
    )

    parser.add_argument(
        '--model_loading_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--generation_rate_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--decoding_strategy_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--encoding_memory_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--logits_calculation_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--binoculars_model_benchmark',
        required=False,
        default='False',
        help='Specify path to experiment config file to run benchmark',
        metavar='CONFIG_FILE_PATH'
    )

    parser.add_argument(
        '--resume',
        required=False,
        choices=['True', 'False'],
        default='False',
        help='Wether or not to resume prior run',
        metavar='BOOL'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def start_logger(logfile_name):
    '''Sets up logging, returns logger'''

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/*.log*'):
            os.remove(file)

    # Create logger
    logger = logging.getLogger('llama3')
    logger.setLevel(config.LOG_LEVEL)

    handler = RotatingFileHandler(
        f'{config.LOG_PATH}/{logfile_name}',
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
