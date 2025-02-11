'''Functions for logging from multiprocessing pool workers. Adapted
from python.org logging cookbook'''

# Standard library imports
import os
import glob
import logging
from typing import Callable

# Internal imports
import configuration as config # pylint: disable=import-error


def configure_listener(logfile: str) -> None:
    '''Function to configure log listener process.'''

    # Clear old log files
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{logfile}*'):
            os.remove(file)

    root=logging.getLogger()
    handler=logging.handlers.RotatingFileHandler(
        logfile,
        maxBytes=config.LOGFILE_SIZE,
        backupCount=config.MAX_LOGFILES
    )
    formatter=logging.Formatter(config.LOG_PREFIX)
    handler.setFormatter(formatter)
    root.addHandler(handler)


def listener_process(queue: Callable, configurer: Callable, logfile: str) -> None:
    '''Loop for listener, gets log records from queue and logs them.'''

    # Set the configuration
    configurer(logfile)

    # Loop until we receive the sentinel 'None' value from the queue
    while True:

        # Get the next log record
        record=queue.get()

        # Break on the sentinel
        if record is None:
            break

        # Otherwise log the record
        logger=logging.getLogger(record.name)
        logger.handle(record)


def configure_worker(queue: Callable) -> None:
    '''Function to configure logging in worker processes.'''

    # Add handler for logging queue
    handler=logging.handlers.QueueHandler(queue)
    root=logging.getLogger()
    root.addHandler(handler)

    # Set log level from configuration file
    root.setLevel(config.LOG_LEVEL)
