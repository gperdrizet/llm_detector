'''Collection of function to run benchmarks'''

from __future__ import annotations
from typing import List, Callable

import os
import logging
from multiprocessing import Manager, Process

import benchmarking.classes.llm as llm_class
import benchmarking.classes.experiment as experiment_class
import benchmarking.functions.benchmarks as benchmark_funcs
import benchmarking.functions.helper as helper_funcs

# Comment ##############################################################
# Code ########################################################################

def run(
        experiment_config_file: str = None,
        resume: bool = False
) -> None:

    '''Generalized function for benchmarking experiments'''

    # Start the logger for this benchmark, using the name of the
    # experiment configuration file
    config_file_basename = os.path.basename(experiment_config_file)
    experiment_name = config_file_basename.split('.')[0]
    logfile_name = f'{experiment_name}.log'

    logger = helper_funcs.start_logger(
            logfile_name = logfile_name,
            logger_name = experiment_name
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
                experiment_name = experiment.experiment_name,
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
        experiment.extend_results(list(results))
        experiment.save_results()
        results = manager.list()

    logger.info('%s run complete', experiment.experiment_name)


def run_batch(
        experiment_name: str = None,
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
    logger = logging.getLogger(f'{experiment_name}.run_batch')

    # Empty holder for result
    result = {}

    # Set the benchmark function to run
    benchmark_func = getattr(benchmark_funcs, benchmark_func)

    # Prepare and load the LLM(s). since all of the runs in a
    # batch are the same except for the iteration number, we can
    # use the first run from the batch as the source for our LLM
    # parameters
    llms = instantiate_llms(run_dict = batch[0])

    # Load each LLM, catching runtime errors
    try:
        for i, _ in enumerate(llms):
            logger.debug('Loading LLM %s', i)
            llms[i].load()

    # Handel the error by constructing a result for this batch
    # with the independent variables populated from the run
    # dictionary list and the dependent variables filled with an
    # appropriate error string. Then call off the batch run
    # by returning to the main process
    except RuntimeError as runtime_error:

        # Log the error
        logger.error(runtime_error)

        results = helper_funcs.handle_model_load_runtime_error(
            runtime_error = runtime_error,
            batch = batch,
            independent_vars = independent_vars,
            dependent_vars = dependent_vars,
            results = results,
            result = result
        )

        return

    # Loop on the conditions in this batch
    for i, run_dict in enumerate(batch, start = 1):

        # Fence to catch runtime error during the run
        try:
            # Call the run specific benchmark function, catching runtime errors
            result = benchmark_func(
                run_dict = run_dict,
                llms = llms,
                data = data
            )

        # Handel the error by constructing a result for this run
        # with the independent variables populated from the run
        # dictionary and the dependent variables filled with an
        # appropriate error string
        except RuntimeError as runtime_error:

            # Log the error
            logger.error(runtime_error)

            result = helper_funcs.handle_benchmark_runtime_error(
                runtime_error = runtime_error,
                run_dict = run_dict,
                independent_vars = independent_vars,
                dependent_vars = dependent_vars,
                result = result
            )

        # Add to results
        results.append(result)
        logger.info('Finished run %s of %s', i, len(batch))

    return

def instantiate_llms(run_dict: dict = None) -> List[Callable]:
    '''Handles instantiating LLM(s)'''

    # Get the logger
    logger = logging.getLogger('benchmarking.instantiate_llms')

    # Instantiate a LLM class instance for the LLM(s), need to
    # handle this one of two ways - if we are loading more than
    # one LLM, the value of hf_model_string will be a list of
    # models to load, if we are only loading one model, it will
    # be a single string. So check the type to make sure we are
    # doing the right thing
    llms = []

    if isinstance(run_dict['hf_model_string'], str):
        llms.append(llm_class.Llm())

    elif isinstance(run_dict['hf_model_string'], list):
        for _ in range(len(run_dict['hf_model_string'])):
            llms.append(llm_class.Llm())

    logger.debug('Instantiated %s LLMs', len(llms))
    logger.debug('Setting LLM parameters from batch: %s', run_dict)

    # Set the LLM parameters
    for parameter_name, value in run_dict.items():

        # Check if this parameter is an attribute of the LLM class
        if parameter_name in dir(llms[0]):

            # Need to handle setting parameters in one of two ways here
            # 1. If we are only running one LLM, the parameter value
            # should be a single string and we can just set it directly.
            # 2. If we are running more than one LLM, the parameter
            # value will be a list of strings and we need to set each
            # value in the list in the corresponding LLM. Check
            # both the number of LLMs and the type of the parameter
            # value to be sure we are doing the right thing

            if len(llms) == 1 and isinstance(value, str) is True:
                logger.debug('Setting %s to %s for LLM', parameter_name, value)
                setattr(llms[0], parameter_name, value)

            elif len(llms) > 1 and isinstance(value, list) is True:

                for i, _ in enumerate(llms):
                    logger.debug('Setting %s to %s for LLM %s', parameter_name, value[i], i)
                    setattr(llms[i], parameter_name, value[i])

    return llms