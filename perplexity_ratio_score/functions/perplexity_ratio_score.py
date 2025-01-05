'''Functions to compute perplexity ratio score for semantically split and sharded text data.'''
from __future__ import annotations

# Standard library imports
import os
import time
import glob
import logging
import multiprocessing as mp
from pathlib import Path

# PyPI imports
import torch
import numpy as np
import pandas as pd

# Internal imports
import classes.llm as llm_class # pylint: disable=import-error
import functions.multiprocess_logging as log_funcs # pylint: disable=import-error
import configuration as config # pylint: disable=import-error
from functions.metrics import perplexity, entropy # pylint: disable=import-error

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

    # Start multiprocess logging
    logger, log_listener, logging_queue=start_logging()

    # Get the list of input files that still need to be scored
    inputs_to_score=read_input_files()

    # Construct GPU worker device list
    gpu_workers=get_gpu_workers()

    # Decide how many fragments to break each input file into
    num_fragments=10000

    # Start the multiprocessing manager
    mp_manager=mp.Manager()

    # Set-up multiprocessing queues for input and output
    input_queue=mp_manager.Queue(-1)
    output_queue=mp_manager.Queue(-1)

    # Set-up a process to collect results from workers
    result_collector=mp.Process(
        target=collect_results,
        args=(
            output_queue,
            len(gpu_workers),
            num_fragments
        )
    )

    logger.info('initialized result collector process')
    result_collector.start()
    logger.info('Started result collector process')

    # Set-up scoring worker processes
    scoring_workers=[]

    for i, gpu in enumerate(gpu_workers):

        scoring_workers.append(
            mp.Process(target=score_shard, args=(input_queue,output_queue,gpu,i,))
        )

        logger.info('Initialized scoring worker %s on GPU %s', i, gpu)

    # Start the score workers
    for i, worker in enumerate(scoring_workers):
        worker.start()
        logger.info('Started scoring worker %s', i)

    # Loop on the input files, fragment them and send the fragments
    # to the workers
    for input_file in inputs_to_score:

        # Get just the filename from the input file path
        input_file_name=os.path.basename(input_file)

        # Use the input file name to create the output file path
        output_file=f'{config.SCORED_DATA_PATH}/{input_file_name}'

        # Check to see if the output file exists
        if Path(output_file).is_file():
            logger.info('Output for %s exists', input_file_name)

            # Read the output data and get the row count
            output_df=pd.read_parquet(output_file)
            output_rows=len(output_df)

            # Use the length of the output dataframe as the start index
            # from which to load the input data
            start_index=output_rows

            logger.info('Output %s contains %s rows',input_file_name,output_rows)

        # If the output file does not exist, load the input from index 0
        else:
            start_index=0
            logger.info('No output for %s exists', input_file_name)

        # Load the input data
        data_df=pd.read_parquet(input_file)

        # Trim the input data using the start index from the end of the output file
        data_df=data_df.iloc[start_index:]

        # Split the data into chunks of 100 rows
        data_df_chunks=np.array_split(data_df, num_fragments)

        # Put each chunk and it's corresponding file name into the queue
        while len(data_df_chunks) > 0:

            # If we have less than two workunits per worker in the queue
            # add another one
            if input_queue.qsize() < len(gpu_workers) * 2:
                input_queue.put([input_file_name, data_df_chunks.pop()])

            # Wait before checking again
            time.sleep(1)

    # Then, send each score worker a done signal
    for i in range(gpu_workers):
        input_queue.put(['Done', None])

    # Join and then close each score worker process
    for worker in scoring_workers:
        worker.join()
        worker.close()

    # Join and close the result collector
    result_collector.join()
    result_collector.close()

    # Clean up the logging process
    logging_queue.put_nowait(None)
    log_listener.join()

    return


def score_shard(
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        gpu: str,
        worker_num: int
) -> None:

    '''Worker function to score chunk of sharded text file.
    sends result to collection worker.'''

    # Set-up worker's logging
    logger=logging.getLogger(f'{__name__}.score_shard')
    logger.info('Worker %s started', worker_num)

    # Model loading specification
    reader_model_string='tiiuae/falcon-7b'
    writer_model_string='tiiuae/falcon-7b-instruct'
    cpu_cores=2

    # Load the models
    reader_model=llm_class.Llm(
        hf_model_string=reader_model_string,
        device_map=gpu,
        cache_dir='/home/siderealyear/projects/huggingface_cache',
        cpu_cores=cpu_cores
    )

    writer_model=llm_class.Llm(
        hf_model_string=writer_model_string,
        device_map=gpu,
        cache_dir='/home/siderealyear/projects/huggingface_cache',
        cpu_cores=cpu_cores
    )

    reader_model.load()
    logger.info(
        'Worker %s reader loaded %s on %s',
        worker_num,
        reader_model_string,
        gpu
    )

    writer_model.load()
    logger.info(
        'Worker %s writer loaded %s on %s',
        worker_num,
        writer_model_string,
        gpu
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

        # Get the next workunit from the queue
        workunit=input_queue.get()

        # Check for 'Done' signal
        if workunit[0] == 'Done':
            output_queue.put(['Done', None])
            logger.info('Worker %s received stop signal', worker_num)
            return

        # Unpack the workunit
        input_file_name=workunit[0]
        data_df=workunit[1]
        data_df.reset_index(inplace=True, drop=True)

        logger.info(
            'Worker %s received workunit with %s rows from %s',
            worker_num,
            len(data_df),
            input_file_name
        )

        # Loop on the dataframe rows, scoring each text and collecting the
        # scores in a list so that we can add them as a new column later
        scores=[]

        for _, row in data_df.iterrows():

            # Fence to catch CUDA OOM
            try:

                # Encode the text with the reader's tokenizer
                encodings=reader_model.tokenizer(
                    row['text'],
                    return_tensors='pt',
                    return_token_type_ids=False
                ).to(gpu)

                # Get reader's logits
                reader_logits=reader_model.model(**encodings).logits

                # Get the writer's logits
                writer_logits=writer_model.model(**encodings).logits

                # Calculate perplexity using the reader's encoding and the writer's logits
                ppl=perplexity(encodings, writer_logits)

                # Calculate the cross-perplexity
                x_ppl=entropy(
                    reader_logits,
                    writer_logits.to(gpu),
                    encodings,
                    reader_model.tokenizer.pad_token_id
                )

                # Finally, get the perplexity ratio score
                score=ppl / x_ppl
                scores.append(score[0])

            except RuntimeError as runtime_error:

                logger.error('Worker %s', runtime_error)

                # For out of memory enter OOM
                if 'CUDA out of memory' in str(runtime_error):
                    error_string = 'OOM'

                # Otherwise enter NAN:
                else:
                    error_string = 'NAN'

                scores.append(error_string)

        # Add the perplexity ratio scores back to the dataframe as a new column
        data_df['perplexity_ratio_score']=scores

        # Put the result in the output queue along with it's corresponding file name
        output_queue.put([input_file_name, data_df])


def collect_results(output_queue: mp.Queue, num_scoring_workers: int, num_fragments: int) -> None:
    '''Collects results from workers, concatenates them and saves to file'''

    # Get function logger
    logger=logging.getLogger(f'{__name__}.collect_results')

    # Main loop to collect results
    results=[]
    done_count=0

    # Start the timer
    start_time=time.time()

    while True:

        # Get the next output from the queue
        output=output_queue.get()

        # Check for stop signal from workers
        if output[0] == 'Done':
            done_count+=1
            if done_count == num_scoring_workers:
                return

        # Unpack the output
        output_file_name=output[0]
        data_df=output[1]
        logger.info('Got result fragment from %s', output_file_name)

        # Add the new data to the results and concatenate
        results.append(data_df)
        results_df=pd.concat(results)
        results_df.reset_index(inplace=True, drop=True)

        # Save the result
        output_file=f'{config.SCORED_DATA_PATH}/{output_file_name}'
        results_df.to_parquet(output_file)
        logger.info('Saved %s result %s of %s', output_file_name, len(results), num_fragments)

        # Get the average scoring rate and predicted time remaining
        dt=time.time() - start_time

        scored_fragments=len(results)
        rate=scored_fragments / dt

        fragments_remaining=num_fragments - scored_fragments
        time_remaining=(fragments_remaining / rate) / (60*60)
        percent_complete=(scored_fragments / num_fragments) * 100

        logger.info('%s %s %% complete',output_file_name,round(percent_complete,1))
        logger.info('Scoring rate: %s fragments per hour',round(rate*60*60,1))
        logger.info('Estimated time remaining: %s hours',int(time_remaining))

        # If we are finished with this input file, clear the results and reset the timer
        if len(results) == num_fragments:
            results=[]
            start_time=time.time()

        # Wait one second before checking the queue again
        time.sleep(1)


def start_logging() -> tuple[logging.Logger, mp.Process, mp.Queue]:
    '''Function to set-up and start multiprocess logger in the main process.'''

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

    return logger, log_listener, logging_queue

def read_input_files() -> list:
    '''Reads list of input files, check against pre-existing output files
    if any, returns list of input files that don't have finished output'''

    # Get function logger
    logger=logging.getLogger(f'{__name__}.read_input_files')

    # Set-up output directory and get files, if any
    Path(config.SCORED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Get list of input files
    input_files=glob.glob(f'{config.INTERMEDIATE_DATA_PATH}/*chunks.*.parquet')
    logger.info('Read %s input files', len(input_files))

    # Loop on the input file list and compare to output file list. Assemble list of
    # Input files that have not been scored or are incomplete
    inputs_to_score=[]

    for i, input_file in enumerate(input_files):

        # Get the input file's name
        input_file_name=os.path.basename(input_file)
        logger.info('Input %s: %s', i+1, input_file_name)

        # Check to see if a corresponding output file exists
        output_file=f'{config.SCORED_DATA_PATH}/{input_file_name}'

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

    return inputs_to_score


def get_gpu_workers() -> list:
    '''Constructs list of GPU worker devices'''

    # Get function logger
    logger=logging.getLogger(f'{__name__}.get_gpu_workers')

    # Check how many GPUs we have
    gpus=torch.cuda.device_count()
    logger.info('Have %s GPUs', gpus)

    # Set the number of workers to assign to each GPU
    workers_per_gpu=1

    # Make a list of the avalible GPU workers
    gpu_workers=[]

    for gpu in range(gpus):
        for _ in range(workers_per_gpu):
            gpu_workers.append(f'cuda:{gpu}')

    logger.info('GPU workers %s', gpu_workers)

    return gpu_workers
