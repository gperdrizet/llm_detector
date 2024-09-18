'''Collection of helper functions.'''
from __future__ import annotations
from typing import Callable, List

import os
import glob
import argparse
import logging
import tracemalloc
from logging.handlers import RotatingFileHandler
import torch
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
        '--perplexity-ratio',
        required = False,
        choices = ['True', 'False'],
        default = 'False',
        help = 'Run perplexity ratio score calculation on Hans 2024 datasets',
        metavar = '<BOOL>'
    )

    parser.add_argument(
        '--perplexity-ratio-v2',
        required = False,
        help = 'Run updated perplexity ratio score calculation on Hans 2024 datasets',
        metavar = '<OUTPUT_FILE_NAME>'
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

def start_memory_tracking(device_map: str = None) -> None:
    '''Starts memory tracking using the device map to pick the right tool'''

    # Start memory tracking using the correct strategy based on device map
    if device_map != 'cpu':

        # Reset memory stats for all GPUs
        for device in config.AVAILABLE_GPUS:
            torch.cuda.reset_peak_memory_stats(device = device)

    elif device_map == 'cpu':
        tracemalloc.start()

def get_peak_memory(device_map: str = None) -> float:
    '''Returns peak memory using the device map to pick the right tool'''

    # Get peak memory using the correct strategy based on device map
    if device_map != 'cpu':
        peak_memory = 0

        for device in config.AVAILABLE_GPUS:
            device_peak_memory = torch.cuda.max_memory_allocated(device = device)
            device_peak_memory = device_peak_memory / (10 ** 9)
            peak_memory += device_peak_memory

    elif device_map == 'cpu':

        _, peak_memory = tracemalloc.get_traced_memory()
        peak_memory = peak_memory / (10 ** 6)
        tracemalloc.stop()

    return peak_memory

def handle_model_load_runtime_error(
        runtime_error: str = None,
        batch: List[dict] = None,
        independent_vars: list = None,
        dependent_vars: list = None,
        results = None,
        result = None
):

    '''Catches runtime errors the occur during model loading.
    Constructs a results dictionary list containing batch's
    values for independent variable and an error string for
    independent variables'''

    # For out of memory enter OOM
    if 'CUDA out of memory' in str(runtime_error):
        error_string='OOM'

    # For anything else, use NAN
    else:
        error_string='NAN'

    # Loop on the conditions in this batch
    for run_dict in batch:

        # Loop on the independent variables and add the value from this
        # run to the result
        for independent_var in independent_vars:
            result[independent_var] = run_dict[independent_var]

        # Enter the error string in all of the dependent variables
        # for this run
        for dependent_var in dependent_vars:
            result[dependent_var] = error_string

        # Add the run result to the results list
        results.append(result)

    return results


def handle_benchmark_runtime_error(
        runtime_error: str = None,
        run_dict: dict = None,
        independent_vars: list = None,
        dependent_vars: list = None,
        result = None
) -> dict:

    '''Catches runtime errors the occur during the benchmark
    run. Constructs and returns a results dictionary
    containing the run's values for independent variable 
    and an error string for independent variables'''

    # For out of memory enter OOM
    if 'CUDA out of memory' in str(runtime_error):
        error_string='OOM'

    # For anything else, use NAN
    else:
        error_string='NAN'

    # Loop on the independent variables and add the value from this
    # run to the result
    for independent_var in independent_vars:
        result[independent_var] = run_dict[independent_var]

    # Enter the error string in all of the dependent variables
    # for this run
    for dependent_var in dependent_vars:
        result[dependent_var] = error_string

    return result

