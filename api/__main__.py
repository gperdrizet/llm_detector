'''Main module to initialize LLMs, set-up and launch Celery & Flask apps
using either Gunicorn or the Flask development server'''

<<<<<<< HEAD:llm_detector_api/__main__.py
import llm_detector_api.functions.flask_app as app_funcs
import llm_detector_api.functions.helper as helper_funcs
import llm_detector_api.configuration as config
=======
import api.functions.flask_app as app_funcs
import api.functions.helper as helper_funcs
>>>>>>> main:api/__main__.py

# Start the logger
logger = helper_funcs.start_logger()

<<<<<<< HEAD:llm_detector_api/__main__.py
logger.info('Running in %s mode', config.MODE)

if config.MODE == 'testing':
    
    # Don't load the LLMs
    observer_model = None
    performer_model = None

elif config.MODE == 'production':

    # Start the models
    observer_model, performer_model=helper_funcs.start_models(logger)
    logger.info('Models started')
=======
# Start the models
reader_model, writer_model = helper_funcs.start_models(logger)
logger.info('Models started')
>>>>>>> main:api/__main__.py

# Initialize Flask app
flask_app=app_funcs.create_flask_celery_app(reader_model, writer_model)
logger.info('Flask app initialized')

# Start the celery app
helper_funcs.start_celery(flask_app, logger)
logger.info('Celery app MainProcess thread started')

# If main is being executed directly, i.e. via: python -m llm_detector
# run the Flask app in a thread using the development server
if __name__ == '__main__':

    # Start the flask app
    helper_funcs.start_flask(flask_app, logger)
    logger.info('Flask app thread started')
