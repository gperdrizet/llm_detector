'''Functions to compute perplexity ratio score for semantically split and sharded text data.'''

# Standard library imports
import os
import glob
import logging
import multiprocessing as mp
from pathlib import Path

# PyPI imports
import torch
import pandas as pd

# Internal imports
import multiprocess_logging as log_funcs

def run(
        log_path: str,
        intermediate_data_path: str,
        scored_data_path: str
) -> None:

    '''Main function to run perplexity ratio scoring.'''

    # Plan here is to build this parallelized and with the ability to
    # restart from the get-go. This way we can scale to more GPUs if
    # we want/need to and/or submit it to Google Cloud as a interruptable batch
    # job. Workflow will be as follows:
    #
    # 1. Glob input file list - this will contain both training and testing
    #  shards of semantically split text in parquet format.
    # 2. Check for corresponding output files for each input, if output exists,
    #  count the lines and compare to input to determine if it is complete.
    # 3. Loop on input files with missing or incomplete outputs, submitting to
    #   n scoring workers, dependent on gpus/memory available.
    # 4. Worker checks for output file, if it exists, reads the data counting
    #  rows so it can resume from the right place.
    # 5. Worker scores text fragments, collecting results.
    # 6. Worker creates or appends to output file.

    # Set-up multiprocess logging to file
    logfile=f'{log_path}/{__name__}.log'
    print(f'Will log to: {logfile}\n')

    logging_queue=mp.Manager().Queue(-1)

    log_listener=mp.Process(
        target=log_funcs.listener_process,
        args=(logging_queue, log_funcs.configure_listener, logfile)
    )

    log_listener.start()

    # Get logger for main process
    log_funcs.configure_worker(logging_queue)
    logger=logging.getLogger(f'{__name__}.run')
    logger.info('Main process started')

    # Get list of input files
    input_files=glob.glob(f'{intermediate_data_path}/*chunks.*.parquet')
    logger.info('Read %s input files', len(input_files))

    # Set-up output directory and get files, if any
    output_directory=scored_data_path
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    output_files=glob.glob(f'{output_directory}/*chunks.*.parquet')
    logger.info('Read %s output files', len(output_files))

    # Loop on the input file list and compare to output file list. Assemble list of
    # Input files that have not been scored or are incomplete
    inputs_to_score=[]

    for i, input_file in enumerate(input_files):

        # Get the input file's name
        input_file_name=os.path.basename(input_file)
        logger.info('Input %s: %s', i+1, input_file_name)

        # Check to see if a corresponding output file exists
        output_file=f'{output_directory}/{input_file_name}'

        if Path(output_file).is_file():
            logger.info('Output exists')

            # Read the output and compare the number of lines to that
            # in the input. If the line numbers don't match, the output
            # is incomplete, add the input to the 'to score' list.
            input_data=pd.read_parquet(input_file)
            output_data=pd.read_parquet(output_file)

            if len(input_data) == len(output_data):
                logger.info('Output is complete')

            else:
                inputs_to_score.append(input_file)
                logger.info('Output is incomplete')

        else:
            inputs_to_score.append(input_file)
            logger.info('No output exists')

    logger.info('Have %s input files to score', len(inputs_to_score))

    # Start the multiprocessing manager
    mp_manager=mp.Manager()

    # Set-up multiprocessing queue to feed workers
    input_queue=mp_manager.Queue(maxsize=len(inputs_to_score))

    # Set-up scoring worker processes
    num_workers=1
    scoring_workers=[]

    for i in range(num_workers):

        scoring_workers.append(
            mp.Process(target=score_shard, args=(input_queue,i,))
        )

        logger.info('initialized worker %s', i)

    # Add the input files to the queue
    for input_file in input_files:
        input_queue.put(input_file)

    logger.info('Input queue loaded')

    # Start the score workers
    for i, worker in enumerate(scoring_workers):
        worker.start()
        logger.info('Started scoring worker %s', i)

    # Then, send each score worker a done signal
    for i in range(num_workers):
        input_queue.put('Done')

    # Join and then close each score worker process
    for worker in scoring_workers:
        worker.join()
        worker.close()

    # Clean up the logging process
    logging_queue.put_nowait(None)
    log_listener.join()

    return


def score_shard(input_queue: mp.Queue, worker_num: int) -> None:
    '''Worker function to score sharded text file.'''

    # Set-up worker's logging
    logger=logging.getLogger(f'{__name__}.score_shard')
    logger.info('Worker %s started', worker_num)

    # Start the main loop
    while True:

        # Get the next file from the queue
        input_file=input_queue.get()

        # Check for 'Done' signal
        if input_file != 'Done':
            logger.info('Worker %s got %s from queue', worker_num, os.path.basename(input_file))

        else:
            logger.info('Worker %s received stop signal', worker_num)
            return
