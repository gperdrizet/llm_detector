'''Main module to initialize LLMs, set-up and launch Celery-Flask app.'''

from threading import Thread
import llm_detector.configuration as config
import llm_detector.classes.llm as llm_class
import llm_detector.functions.flask_app as app_funcs
import llm_detector.functions.helper as helper_funcs

def create_app():
    '''Experimental function to run llm detector as Gunicorn app'''

    # Start logger
    logger=helper_funcs.start_logger()

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

    # Initialize Flask app
    flask_app=app_funcs.create_flask_celery_app(observer_model, performer_model)
    logger.info('Flask app initialized')

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

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=flask_app.run,
        kwargs={'host':config.IP_ADDRESS,'port':config.PORT}
    )

    logger.info('Flask app thread initialized')

    # Start the flask app thread
    flask_app_thread.start()
    logger.info('Flask app thread started')
