'''Internal LLM detector API.'''

from typing import Callable
import random
from flask import Flask, request
from celery import Celery, Task, shared_task
from celery.app import trace
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
import api.configuration as config
import api.functions.scoring as scoring_funcs
# pylint: disable=W0223

# Comment ##############################################################
# Code ########################################################################

# Disable return portion task success message log so that
# user messages don't get logged.
trace.LOG_SUCCESS = '''\
Task %(name)s[%(id)s] succeeded in %(runtime)ss\
'''

def create_celery_app(app: Flask) -> Celery:
    '''Sets up Celery app object'''

    class FlaskTask(Task):
        '''Gives task function an active Flask context'''

        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    # Create Celery app
    celery_app = Celery(app.name, task_cls = FlaskTask)

    # Add configuration from Flask app's Celery config. dict
    celery_app.config_from_object(app.config['CELERY'])

    # Configure logging
    celery_app.log.setup(
        loglevel = 'INFO',
        logfile = f'{config.LOG_PATH}/celery.log',
        colorize = None
    )

    # Set as default and add to extensions
    celery_app.set_default()
    app.extensions['celery'] = celery_app

    return celery_app

def create_flask_celery_app(
        reader_model: Callable = None,
        writer_model: Callable = None,
        perplexity_ratio_kld_kde: Callable = None,
        tfidf_luts: Callable = None,
        tfidf_kld_kde: Callable = None,
        model: Callable = None
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
    def score_text(
            suspect_string: str = None,
            response_mode: str = 'default'
    ) -> str:

        '''Takes a string and scores it, returns a dict.
        containing the author call and the original string'''

        logger.info('Submitting string for score.')
        logger.info('Response mode is: %s', response_mode)

        # Check to make sure that text is of sane length
        text_length = len(suspect_string.split(' '))

        if text_length < 50 or text_length > 400:

            reply = '''For best results text should be longer than 50 words and\
                  shorter than 400 words.'''

        else:

            # Call the real scoring function or mock based on mode
            if config.MODE == 'testing':

                # Mock the score with a random float
                score = [random.uniform(0, 1)]

                # Threshold the score
                if score[0] >= 0.5:
                    reply = 'Text is human'

                elif score[0] < 0.5:
                    reply = 'Text is synthetic'

            elif config.MODE == 'production':

                # Call the scoring function
                response = scoring_funcs.score_string(
                    reader_model,
                    writer_model,
                    perplexity_ratio_kld_kde,
                    tfidf_luts,
                    tfidf_kld_kde,
                    model,
                    suspect_string,
                    response_mode
                )

                if response_mode == 'default':

                    human_probability = response[0] * 100
                    machine_probability = response[1] * 100

                    if human_probability > machine_probability:
                        reply = f'''{human_probability:.1f}% chance that this text was written by\
                              a human.'''

                    elif human_probability < machine_probability:
                        reply = f'{machine_probability:.1f}% chance that this text was written by a machine.'

                elif response_mode == 'verbose':

                    features = ('Fragment length (tokens): '
                                f"{response[2]['Fragment length (tokens)']:.0f}\n"
                                'Perplexity: '
                                f"{response[2]['Perplexity']:.2f}\n"
                                'Cross-perplexity: '
                                f"{response[2]['Cross-perplexity']:.2f}\n"
                                'Perplexity ratio score: '
                                f"{response[2]['Perplexity ratio score']:.3f}\n"
                                'Perplexity ratio Kullback-Leibler score: '
                                f"{response[2]['Perplexity ratio Kullback-Leibler score']:.3f}\n"
                                'Human TF-IDF: '
                                f"{response[2]['Human TF-IDF']:.2f}\n"
                                'Synthetic TF-IDF: '
                                f"{response[2]['Synthetic TF-IDF']:.2f}\n"
                                'TF-IDF score: '
                                f"{response[2]['TF-IDF score']:.3f}\n"
                                'TF-IDF Kullback-Leibler score: '
                                f"{response[2]['TF-IDF Kullback-Leibler score']:.3f}")

                    reply = f'''Class probabilities: human = {response[0]:.3f},\
                          machine = {response[1]:.3f}\n\nFeature values:\n{features}.'''

        # Return the result from the output queue
        return {'reply': reply, 'text': suspect_string}

    # Set listener for text strings via POST
    @app.post('/submit_text')
    def submit_text() -> dict:
        '''Submits text for scoring. Returns dict. containing 
        result id.'''

        # Get the suspect text string from the request data
        request_data = request.get_json()
        text_string = request_data['string']
        response_mode = request_data['response_mode']

        # Submit the text for scoring
        result = score_text.delay(text_string, response_mode)

        return {'result_id': result.id}

    @app.get('/result/<result_id>')
    def task_result(result_id: str) -> dict:
        '''Gets result by result id. Returns dictionary
        with task status'''

        # Get the result
        result = AsyncResult(result_id)

        # Return status and result if ready
        return {
            'ready': result.ready(),
            'successful': result.successful(),
            'value': result.result if result.ready() else None,
        }

    return app
