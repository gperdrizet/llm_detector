'''Collection of helper functions.'''

from __future__ import annotations
from typing import Callable

import os
import glob
import re
import logging
from threading import Thread
from logging.handlers import RotatingFileHandler
import api.configuration as config
import api.classes.llm as llm_class

# Comment ##############################################################
# Code ########################################################################

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

    formatter = logging.Formatter(
        config.LOG_PREFIX, datefmt='%Y-%m-%d %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('############################################### ')
    logger.info('########### Starting LLM detector ############# ')
    logger.info('############################################### ')

    return logger


def start_models(logger: Callable) -> list[Callable, Callable]:
    '''Initializes, loads and returns the reader and writer LLM'''

    # Configure and load two instances of the model, one base for the
    # reader and one instruct for the writer. Use different GPUs.
    reader_model = llm_class.Llm(
        hf_model_string = config.READER_MODEL,
        device_map = config.READER_DEVICE,
        logger = logger
    )

    reader_model.load()
    logger.info('Loaded reader model')

    writer_model = llm_class.Llm(
        hf_model_string = config.WRITER_MODEL,
        device_map = config.WRITER_DEVICE,
        logger = logger
    )

    writer_model.load()
    logger.info('Loaded writer model')

    return reader_model, writer_model


def start_celery(flask_app: Callable, logger: Callable) -> None:
    '''Initializes Celery and starts it in a thread'''

    # Get the Celery app
    celery_app = flask_app.extensions['celery']
    logger.info('Celery app initialized')

    # Put the Celery into a thread
    celery_app_thread = Thread(
        target = celery_app.worker_main,
        args = [['worker', '--pool=solo', f'--loglevel={config.LOG_LEVEL}']]
    )

    logger.info('Celery app MainProcess thread initialized')

    # Start the Celery app thread
    celery_app_thread.start()
    logger.info('Celery app MainProcess thread started')


def start_flask(flask_app: Callable, logger: Callable):
    '''Starts flask in a thread via the development server'''

    # Put the flask app into a thread
    flask_app_thread = Thread(
        target = flask_app.run,
        kwargs = {'host':config.HOST_IP,'port':config.FLASK_PORT}
    )

    logger.info('Flask app thread initialized')

    # Start the flask app thread
    flask_app_thread.start()

def clean_text(text: str = None, sw = None, lemmatizer = None) -> str:
    '''Cleans up text string for TF-IDF'''

    # Lowercase everything
    text = text.lower()

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", "",text)

    # Remove html tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'',text)

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

    # Remove punctuations
    for p in punctuations:
        text = text.replace(p,'')

    # Remove stopwords
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)

    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    return text
