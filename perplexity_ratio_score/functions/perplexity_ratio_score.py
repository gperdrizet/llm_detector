'''Functions to compute perplexity ratio score for semantically split and sharded text data.'''

# Standard library imports
import os
import glob
import logging
import threading
import multiprocessing as mp
from pathlib import Path

# PyPI imports
import pandas as pd

# Internal imports
import perplexity_ratio_score.functions.multiprocess_logging as log_funcs
import perplexity_ratio_score.configuration as config

def run():
    '''Main function to run perplexity ratio scoring.'''

    # Plan here is to build this parallelized and with the ability to
    # restart from the get-go. This way we can scale to more GPUs if
    # we want/need to and/or submitt it to Google Cloud as a interutable batch
    # job. Workflow will be as follows:
    #
    # 1. Glob input file list - this will contain both training and testing
    #  shards of semantically split text in parquet format.
    # 2. Check for corresponding output files for each input, if output exists,
    #  count the lines and compare to input to determin if it is complete.
    # 3. Loop on input files with missing or incomplete outputs, submitting to 
    #   n scoring workers, dependent on gpus/memory avalible.
    # 4. Worker checks for output file, if it exists, reads the data counting
    #  rows so it can resume from the right place.
    # 5. Worker scores text fragments, collecting results.
    # 6. Worker creates or appends to output file.

    # Set-up multiprocess logging to file
    logfile=f'{config.LOG_PATH}/{__name__}.log'
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
    logger.info(f'Main process started')

    # Get list of input files
    input_files=glob.glob(f'{config.INTERMEDIATE_DATA_PATH}/*chunks.*.parquet')
    logger.info(f'Read {len(input_files)} input files')

    # Set-up output directory and get files, if any
    output_directory=config.SCORED_DATA_PATH
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    output_files=glob.glob(f'{output_directory}/*chunks.*.parquet')
    logger.info(f'Read {len(output_files)} output files')

    # Loop on the input file list and compare to output file list. Assemble list of
    # Input files that have not been scored or are incomplete
    inputs_to_score=[]

    for i, input_file in enumerate(input_files):

        # Get the input file's name
        input_file_name=os.path.basename(input_file)
        logger.info(f'Input {i+1}: {input_file_name}')

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

    logger.info(f'Have {len(inputs_to_score)} input files to score')

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

        logger.info(f'Initalized worker {i}')

    # Add the input files to the queue
    for input_file in input_files:
        input_queue.put(input_file)

    logger.info(f'Input queue loaded')

    # Start the score workers
    for i, worker in enumerate(scoring_workers):
        worker.start()
        logger.info(f'Started scoring worker {i}')

    # Then, send each score worker a done signial
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
    logger.info(f'Worker {worker_num} started')

    # Start the main loop
    while True:

        # Get the next file from the queue
        input_file=input_queue.get()

        # Check for 'Done' signal
        if input_file != 'Done':
            logger.info(f'Worker {worker_num} got {os.path.basename(input_file)} from queue')

        else:
            logger.info(f'Worker {worker_num} recived stop signal')
            return