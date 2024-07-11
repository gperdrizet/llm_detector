'''Internal LLM detector API.'''

from typing import Callable
from flask import Flask, request # type: ignore
from celery import Celery, Task, shared_task # type: ignore
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
import api.configuration as config
import api.functions.scoring as scoring_funcs
# pylint: disable=W0223

def create_celery_app(app: Flask) -> Celery:
    '''Sets up Celery app object'''

    class FlaskTask(Task):
        '''Gives task function an active Flask context'''

        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    # Create Celery app
    celery_app = Celery(app.name, task_cls=FlaskTask)

    # Add configuration from Flask app's Celery config. dict
    celery_app.config_from_object(app.config["CELERY"])

    # Configure logging
    celery_app.log.setup(
        loglevel = 'INFO',
        logfile = f'{config.LOG_PATH}/celery.log',
        colorize = None
    )

    # Set as default and add to extensions
    celery_app.set_default()
    app.extensions["celery"] = celery_app

    return celery_app

def create_flask_celery_app(
        observer_model: Callable,
        performer_model: Callable
) -> Flask:

    '''Creates Flask app for use with Celery'''

    # Make the app
    app = Flask(__name__)

    # Set the Celery configuration
    app.config.from_mapping(
        CELERY = dict(
            broker_url = config.REDIS_URL,
            result_backend = config.REDIS_URL,
            task_ignore_result = True,
            broker_connection_retry_on_startup = True
        ),
    )

    app.config.from_prefixed_env()

    # Make the celery app
    create_celery_app(app)

    # Get task logger
    logger = get_task_logger(__name__)

    @shared_task(ignore_result = False)
    def score_text(suspect_string: str) -> str:
        '''Submits text string for scoring'''

        logger.info(f'Submitting for score: {suspect_string}')

        # Call the scoring function
        score = scoring_funcs.score_string(
            observer_model,
            performer_model,
            suspect_string
        )

        # Return the result from the output queue
        return {'score': score[0], 'text': suspect_string}

    # Set listener for text strings via POST
    @app.post('/submit_text')
    def submit_text() -> dict:
        '''Submits text for scoring. Returns dict. containing 
        result id.'''

        # Get the suspect text string from the request data
        request_data = request.get_json()
        text_string = request_data['string']

        # Submit the text for scoring
        result = score_text.delay(text_string)

        return {"result_id": result.id}

    @app.get("/result/<result_id>")
    def task_result(result_id: str) -> dict:
        '''Gets result by result id. Returns dictionary
        with task status'''

        # Get the result
        result = AsyncResult(result_id)

        # Return status and result if ready
        return {
            "ready": result.ready(),
            "successful": result.successful(),
            "value": result.result if result.ready() else None,
        }

    return app
