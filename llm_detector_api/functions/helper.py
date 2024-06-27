'''Collection of helper functions.'''

from __future__ import annotations
from typing import Callable
import os
import glob
import logging
from threading import Thread
from logging.handlers import RotatingFileHandler
import llm_detector_api.configuration as config
import llm_detector_api.classes.llm as llm_class

def start_logger() -> Callable:
    '''Sets up logging, returns logger'''

    # Clear logs if asked
    if config.CLEAR_LOGS is True:
        for file in glob.glob(f'{config.LOG_PATH}/*.log*'):
            os.remove(file)

    # Create logger
    logger = logging.getLogger('llm_detector')
    logger.setLevel(config.LOG_LEVEL)

    handler = RotatingFileHandler(
        f'{config.LOG_PATH}/llm_detector.log',
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


def start_models(logger: Callable) -> list[Callable, Callable]:
    '''Initializes, loads and returns the observer and performer LLM'''

    # Configure and load two instances of the model, one base for the observer
    # and one instruct for the performer. Use different GPUs.
    observer_model=llm_class.Llm(
        hf_model_string=config.OBSERVER_MODEL,
        device_map=config.OBSERVER_DEVICE,
        logger=logger
    )

    observer_model.load()
    logger.info('Loaded observer model')

    performer_model=llm_class.Llm(
        hf_model_string=config.PERFORMER_MODEL,
        device_map=config.PERFORMER_DEVICE,
        logger=logger
    )

    performer_model.load()
    logger.info('Loaded performer model')

    return observer_model, performer_model


def start_celery(flask_app: Callable, logger: Callable) -> None:
    '''Initializes Celery and starts it in a thread'''

    # Get the Celery app
    celery_app=flask_app.extensions['celery']
    logger.info('Celery app initialized')

    # Put the Celery into a thread
    celery_app_thread=Thread(
        target=celery_app.worker_main,
        args=[['worker', '--pool=solo', f'--loglevel={config.LOG_LEVEL}']]
    )

    logger.info('Celery app MainProcess thread initialized')

    # Start the Celery app thread
    celery_app_thread.start()
    logger.info('Celery app MainProcess thread started')


def start_flask(flask_app: Callable, logger: Callable):
    '''Starts flask in a thread via the development server'''

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=flask_app.run,
        kwargs={'host':config.IP_ADDRESS,'port':config.PORT}
    )

    logger.info('Flask app thread initialized')

    # Start the flask app thread
    flask_app_thread.start()
