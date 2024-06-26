'''Main module to initialize LLMs, set-up and launch Celery & Flask apps
using either Gunicorn or the Flask development server'''

import llm_detector.functions.flask_app as app_funcs
import llm_detector.functions.helper as helper_funcs

# Start the logger
logger=helper_funcs.start_logger()

# Start the models
observer_model, performer_model=helper_funcs.start_models(logger)
logger.info('Models started')

# Initialize Flask app
flask_app=app_funcs.create_flask_celery_app(observer_model, performer_model)
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
