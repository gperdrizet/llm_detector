# '''Functions for logging from multiprocessing pool workers. Adapted
# from python.org logging cookbook'''

# from typing import Callable

# import pathlib
# import logging
# import logging.handlers

# import configuration as config


# def configure_listener(logfile: str) -> None:
#     '''Function to configure log listener process.'''

#     # Clear old logfile
#     pathlib.Path(logfile).unlink(missing_ok=True)

#     root = logging.getLogger()
#     handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=100000, backupCount=10)
#     formatter = logging.Formatter(config.LOG_PREFIX)
#     handler.setFormatter(formatter)
#     root.addHandler(handler)


# def listener_process(queue: Callable, configurer: Callable, logfile: str) -> None:
#     '''Loop for listener, gets log records from queue and logs them.'''

#     # Set the configuration
#     configurer(logfile)
    
#     # Loop until we recive the sentinel 'None' value from the queue
#     while True:

#         # Get the next log record
#         record = queue.get()

#         # Break on the sentinel
#         if record is None:
#             break

#         # Otherwise log the record
#         logger = logging.getLogger(record.name)
#         logger.handle(record)


# def configure_worker(queue: Callable) -> None:
#     '''Function to configure logging in worker processes.'''
    
#     # Add handler for logging queue
#     handler = logging.handlers.QueueHandler(queue)
#     root = logging.getLogger()
#     root.addHandler(handler)

#     # Set log level from configuration file
#     root.setLevel(config.LOG_LEVEL)
