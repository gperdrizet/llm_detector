'''Main module to initialize LLMs, set-up and launch Celery & Flask apps
using either Gunicorn or the Flask development server'''

import pickle
from threading import Thread
import api.functions.flask_app as app_funcs
import api.functions.helper as helper_funcs
import api.configuration as config

# Start the logger
logger = helper_funcs.start_logger()
logger.info('Running in %s mode', config.MODE)

# Draw a bot traffic plot on app start
helper_funcs.update_traffic_plot()

# Schedule the bot traffic plot update in a worker thread
bot_traffic_thread = Thread(
    target = helper_funcs.schedule_traffic_plot_update
)

bot_traffic_thread.start()

if config.MODE == 'testing':

    # Initialize Flask app without llms
    flask_app = app_funcs.create_flask_celery_app(None, None)
    logger.info('Flask app initialized')

elif config.MODE == 'production':

    # Start the models
    reader_model, writer_model = helper_funcs.start_models(logger)
    logger.info('Models started')

    # Load the other scoring assets

    # Load the perplexity ratio Kullback-Leibler kernel density estimate
    with open(config.PERPLEXITY_RATIO_KLD_KDE, 'rb') as input_file:
        perplexity_ratio_kld_kde = pickle.load(input_file)

    # Load the TF-IDF luts
    with open(config.TFIDF_LUT, 'rb') as input_file:
        tfidf_luts = pickle.load(input_file)

    # Load the TF_IDF Kullback-Leibler kernel density estimate
    with open(config.TFIDF_SCORE_KLD_KDE, 'rb') as input_file:
        tfidf_kld_kde = pickle.load(input_file)

    # Load the model
    with open(config.XGBOOST_CLASSIFIER, 'rb') as input_file:
        model = pickle.load(input_file)

    # Initialize Flask app
    flask_app = app_funcs.create_flask_celery_app(
        reader_model,
        writer_model,
        perplexity_ratio_kld_kde,
        tfidf_luts,
        tfidf_kld_kde,
        model
    )

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
