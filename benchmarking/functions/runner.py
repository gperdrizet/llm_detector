'''Collection of function to run benchmarks'''

from __future__ import annotations
from typing import List

import os
import logging
# import time
# import random
# import tracemalloc
# from random import sample

from multiprocessing import Manager, Process

# import torch
# import benchmarking.configuration as config
import benchmarking.classes.llm as llm_class
import benchmarking.classes.experiment as experiment_class
import benchmarking.functions.helper as helper_funcs
import benchmarking.functions.benchmarks as benchmark_funcs

# Comment ##############################################################
# Code ########################################################################

def run(
        experiment_config_file: str = None,
        resume: bool = False
) -> None:

    '''Generalized function for benchmarking experiments'''

    # Start the logger, naming the log file with the same name
    # as the configuration file
    config_file_basename = os.path.basename(experiment_config_file)
    experiment_name = config_file_basename.split('.')[0]
    logfile_name = f'{experiment_name}.log'

    logger = helper_funcs.start_logger(
        logfile_name = logfile_name,
        logger_name = 'benchmarking'
    )

    # Instantiate experiment class instance
    experiment = experiment_class.Experiment(
        experiment_config_file,
        resume
    )

    logger.info('Running %s', experiment.experiment_name)
    logger.info('Experiment class instance created')

    if resume is True:
        logger.info('Resumed data collection from prior run')

    # Start the multiprocessing manager and put a list into
    # shared memory for results
    manager = Manager()
    results = manager.list()

    logger.info('Multiprocessing manager created')
    logger.debug('Shared memory results type: %s', type(results))
    logger.debug('Shared memory results: %s', results)

    # Loop experiment run batches
    logger.info('Will run %s batches of %s runs each',
                           len(experiment.run_batches_list), len(experiment.run_batches_list[0]))

    for i, batch in enumerate(experiment.run_batches_list, start=1):
        logger.info('Running batch %s of %s', i, len(experiment.run_batches_list))

        # Run the batch in a subprocess
        p = Process(target = run_batch,
            kwargs = dict(
                benchmark_func = experiment.benchmark_func,
                batch = batch,
                independent_vars = experiment.independent_vars.keys(),
                dependent_vars = experiment.dependent_vars.keys(),
                data = experiment.data,
                results = results
            )
        )

        # Run the job
        logger.debug('Starting batch job process')
        p.start()
        logger.debug('Batch job process start unblocked')

        logger.debug('Joining batch job process')
        p.join()
        logger.debug('Batch job process join unblocked')
        logger.debug('Result: %s', results)

        # Update and save results, clear shared memory for next batch
        experiment.extend_results(results)
        experiment.save_results()
        results = manager.list()

    logger.info('%s run complete', experiment.experiment_name)


def run_batch(
        benchmark_func: str = None,
        batch: List[dict] = None,
        independent_vars: list = None,
        dependent_vars: list = None,
        data = None,
        results = None
) -> None:

    '''Guts of the main loop on conditions called by the benchmark 
    function. Encapsulated in a function so it can be called in a 
    subprocess to make sure that everything incidental to the run, dies
    with the run.'''

    # Get the logger
    logger = logging.getLogger('benchmarking.run_batch')

    # Empty holder for result
    result = {}

    # Set the benchmark function to run
    benchmark_func = getattr(benchmark_funcs, benchmark_func)

    # Prepare and load the LLM(s). since all of the conditions in a
    # batch are the same except for the iteration number, we can
    # use the first run from the batch as the source for our LLM
    # parameters

    # Instantiate a LLM class instance for the LLM(s)
    llms = []

    for _ in range(len(batch[0]['hf_model_string'])):
        llms.append(llm_class.Llm())

    # Set the LLM parameters
    for parameter_name, values in batch[0].items():

        # Check if this parameter is an attribute of the LLM class
        if parameter_name in dir(llms[0]):

            # If it is, set each value in the corresponding LLM
            for i, _ in enumerate(llms):
                setattr(llms[i], parameter_name, values[i])

    # Load each LLM, catching CUDA errors
    try:
        for i, _ in enumerate(llms):
            llms[i].load()

    # If anything weird happens, we need to skip this batch. Log the
    # error and enter appropriate error string in the dependent
    # variables then return
    except RuntimeError as runtime_error:

        logger.error(runtime_error)

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

        # Then call off the run by returning the results
        return results

    # Loop on the conditions in this batch
    for i, run_dict in enumerate(batch, start=1):

        # Call the run specific benchmark function, catching CUDA errors
        try:
            result = benchmark_func(
                run_dict = run_dict,
                llms = llms,
                data = data
            )

        # If anything weird happens, print the error and enter appropriate
        # error string in the dependent variables
        except RuntimeError as runtime_error:

            logger.error(runtime_error)

            # For out of memory enter OOM
            if 'CUDA out of memory' in str(runtime_error):
                error_string='OOM'

            # For anything else, use NAN
            else:
                error_string='NAN'

            # Enter the error string in all of the dependent variables
            # for this run
            for dependent_var in dependent_vars:
                result[dependent_var] = error_string

        # Then add the values of the independent variables to the returned results
        for independent_var in independent_vars:

            # Record the values in the experiment class instance
            result[independent_var] = run_dict[independent_var]

        results.append(result)

        logger.info('Finished run %s of %s', i, len(batch))
