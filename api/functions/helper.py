'''Collection of helper functions.'''

from __future__ import annotations
from typing import Callable

import os
import glob
import re
import logging
import time
import datetime
from threading import Thread
from logging.handlers import RotatingFileHandler

import schedule
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import configuration as config
import classes.llm as llm_class

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

    formatter=logging.Formatter(
        config.LOG_PREFIX, datefmt = '%Y-%m-%d %I:%M %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('############################################### ')
    logger.info('########### Starting LLM detector ############# ')
    logger.info('############################################### ')

    return logger

def schedule_traffic_plot_update() -> None:
    '''Runs traffic plot update ever six hours centered on midnight'''

    # Call bot traffic plot update every 6 hours
    schedule.every().day.at('00:00').do(update_traffic_plot)
    schedule.every().day.at('06:00').do(update_traffic_plot)
    schedule.every().day.at('12:00').do(update_traffic_plot)
    schedule.every().day.at('18:00').do(update_traffic_plot)

    # Checks whether a scheduled task is pending to run or not
    while True:
        schedule.run_pending()

        # Sleep
        time.sleep(60)


def update_traffic_plot() -> None:
    'Draws and saves updated traffic plot'

    # Load the traffic data
    data_df=pd.read_csv(config.FRAGMENT_TURNAROUND_DATA, header=None)

    # Create and format some new columns
    data_df.rename(
        columns={0: 'Time fragment received', 1: 'Time reply returned'},
        inplace=True
    )

    data_df['Fragment datetime'] = pd.to_datetime(
        data_df['Time fragment received'],
        unit='s'
    )

    data_df['dT']=(data_df['Time reply returned']
                     - data_df['Time fragment received'])

    # Get time now to set x-axis limits
    x_max=time.time()

    # Draw the plots
    fig, axs=plt.subplots(
        2,
        2,
        figsize=(7, 7),
        gridspec_kw={'wspace':0.3, 'hspace':0.4}
    )

    fig.suptitle('Telegram bot traffic and API latency', size='x-large')

    date_fmt=mdates.DateFormatter('%-I %p')

    axs[0,0].set_title('Last 6 hours')
    axs[0,0].scatter(data_df['Fragment datetime'], data_df['dT'])

    x_min=x_max - (6 * 60 * 60)

    axs[0,0].set_xlim(
        pd.to_datetime(x_min, unit='s'),
        pd.to_datetime(x_max, unit='s')
    )

    axs[0,0].set_xticks(axs[0,0].get_xticks().tolist())
    axs[0,0].set_xticklabels(axs[0,0].get_xticks(), rotation=45)
    axs[0,0].xaxis.set_major_formatter(date_fmt)
    axs[0,0].set_ylabel('Turnaround time (s) ')

    axs[0,1].set_title('Last 12 hours')
    axs[0,1].scatter(data_df['Fragment datetime'], data_df['dT'])

    x_min=x_max - (12 * 60 * 60)

    axs[0,1].set_xlim(
        pd.to_datetime(x_min, unit='s'),
        pd.to_datetime(x_max, unit='s')
    )

    axs[0,1].set_xticks(axs[0,1].get_xticks().tolist())
    axs[0,1].set_xticklabels(axs[0,1].get_xticks(), rotation = 45)
    axs[0,1].xaxis.set_major_formatter(date_fmt)
    axs[0,1].set_ylabel('Turnaround time (s) ')

    axs[1,0].set_title('Last 24 hours')
    axs[1,0].scatter(data_df['Fragment datetime'], data_df['dT'])

    x_min=x_max - (24 * 60 * 60)

    axs[1,0].set_xlim(
        pd.to_datetime(x_min, unit='s'),
        pd.to_datetime(x_max, unit='s')
    )

    axs[1,0].set_xticks(axs[1,0].get_xticks().tolist())
    axs[1,0].set_xticklabels(axs[1,0].get_xticks(), rotation = 45)
    axs[1,0].xaxis.set_major_formatter(date_fmt)
    axs[1,0].set_ylabel('Turnaround time (s) ')

    axs[1,1].set_title('All time')
    axs[1,1].scatter(data_df['Fragment datetime'], data_df['dT'])
    axs[1,1].set_xlim(xmax = pd.to_datetime(x_max, unit = 's'))
    axs[1,1].set_xticks(axs[1,1].get_xticks().tolist())
    axs[1,1].set_xticklabels(axs[1,1].get_xticks(), rotation = 45)
    axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axs[1,1].set_ylabel('Turnaround time (s) ')

    # Save the plot
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    fig.savefig(f'{config.LOG_PATH}/{date}_bot_traffic.jpg', dpi = 150)

    return


def start_models(logger: Callable) -> list[Callable, Callable]:
    '''Initializes, loads and returns the reader and writer LLM'''

    # Configure and load two instances of the model, one base for the
    # reader and one instruct for the writer. Use different GPUs.
    reader_model=llm_class.Llm(
        hf_model_string=config.READER_MODEL,
        device_map=config.READER_DEVICE,
        logger=logger
    )

    reader_model.load()
    logger.info('Loaded reader model')

    writer_model=llm_class.Llm(
        hf_model_string=config.WRITER_MODEL,
        device_map=config.WRITER_DEVICE,
        logger=logger
    )

    writer_model.load()
    logger.info('Loaded writer model')

    return reader_model, writer_model


def start_celery(flask_app: Callable, logger: Callable) -> None:
    '''Initializes Celery and starts it in a thread'''

    # Get the Celery app
    celery_app=flask_app.extensions['celery']
    logger.info('Celery app initialized')

    # Put the Celery into a thread
    celery_app_thread=Thread(
        target=celery_app.worker_main,
        args=[['worker','--pool=solo',f'--loglevel={config.LOG_LEVEL}']]
    )

    logger.info('Celery app MainProcess thread initialized')

    # Start the Celery app thread
    celery_app_thread.start()


def start_flask(flask_app: Callable, logger: Callable):
    '''Starts flask in a thread via the development server'''

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=flask_app.run,
        kwargs={'host':config.HOST_IP,'port':config.FLASK_PORT}
    )

    logger.info('Flask app thread initialized')

    # Start the flask app thread
    flask_app_thread.start()

def clean_text(text: str=None, sw=None, lemmatizer=None) -> str:
    '''Cleans up text string for TF-IDF'''

    # Lowercase everything
    text=text.lower()

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text=re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # Remove URLs
    text=re.sub(r"http\S+", "",text)

    # Remove html tags
    html=re.compile(r'<.*?>')
    text=html.sub(r'',text)

    punctuations='@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

    # Remove punctuations
    for p in punctuations:
        text=text.replace(p,'')

    # Remove stopwords
    text=[word.lower() for word in text.split() if word.lower() not in sw]
    text=[lemmatizer.lemmatize(word) for word in text]
    text=" ".join(text)

    # Remove emojis
    emoji_pattern=re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

    text=emoji_pattern.sub(r'', text)

    return text
