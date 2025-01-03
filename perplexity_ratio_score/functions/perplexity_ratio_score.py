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
import classes.llm as llm_class # pylint: disable=import-error
import functions.multiprocess_logging as log_funcs # pylint: disable=import-error
import configuration as config # pylint: disable=import-error

def run() -> None:

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
    logfile=f'{config.LOG_PATH}/{__name__}.log'

    # Make sure we have a logs directory
    Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/{logfile}*'):
            os.remove(file)

    print(f'Will log to: {logfile}\n')

    # Start queue for multiprocess logging
    logging_queue=mp.Manager().Queue(-1)

    # Start log listener process
    log_listener=mp.Process(
        target=log_funcs.listener_process,
        args=(logging_queue, log_funcs.configure_listener, logfile)
    )

    log_listener.start()

    # Get logger for main process
    log_funcs.configure_worker(logging_queue)
    logger=logging.getLogger(f'{__name__}.run')
    logger.info('Main process started')

    # Check how many GPUs we have
    gpus=torch.cuda.device_count()
    logger.info('Have %s GPUs', gpus)

    # Get list of input files
    input_files=glob.glob(f'{config.INTERMEDIATE_DATA_PATH}/*chunks.*.parquet')
    logger.info('Read %s input files', len(input_files))

    # Set-up output directory and get files, if any
    output_directory=config.SCORED_DATA_PATH
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

    # Model loading specification
    reader_model_string='tiiuae/falcon-7b'
    writer_model_string='tiiuae/falcon-7b-instruct'
    reader_device='cuda:0'
    writer_device='cuda:0'
    cpu_cores=12

    # Load the models
    reader_model=llm_class.Llm(
        hf_model_string=reader_model_string,
        device_map=reader_device,
        cache_dir='/home/siderealyear/projects/huggingface_cache',
        cpu_cores=cpu_cores
    )

    writer_model=llm_class.Llm(
        hf_model_string=writer_model_string,
        device_map=writer_device,
        cache_dir='/home/siderealyear/projects/huggingface_cache',
        cpu_cores=cpu_cores
    )

    reader_model.load()
    logger.info(
        'Worker %s reader loaded %s on %s',
        worker_num,
        reader_model_string,
        reader_device
    )

    writer_model.load()
    logger.info(
        'Worker %s writer loaded %s on %s',
        worker_num,
        writer_model_string,
        writer_device
    )

    # Set the models to evaluation mode to deactivate any dropout
    # modules to ensure reproducibility of results during evaluation
    reader_model.model.eval()
    writer_model.model.eval()

    # Add end of sequence for the pad token if one has not been defined
    if not reader_model.tokenizer.pad_token:
        reader_model.tokenizer.pad_token=reader_model.tokenizer.eos_token

    # Start the main loop
    while True:

        # Get the next file from the queue
        input_file=input_queue.get()

        # Check for 'Done' signal
        if input_file == 'Done':
            logger.info('Worker %s received stop signal', worker_num)
            return

        else:

            # Get the input file name
            input_file_name=os.path.basename(input_file)
            logger.info('Worker %s got %s from queue', worker_num, input_file_name)

            # Load the data
            data_df=pd.read_parquet(input_file)
            logger.info('Worker %s: %s has %s rows', worker_num, input_file_name, len(data_df))

