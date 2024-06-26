'''Main module to launch flask app and scoring backend'''

import queue
from threading import Thread
# import torch
# import llm_detector.classes.llm as llm_class
import llm_detector.configuration as config
import llm_detector.functions.flask_app as app_funcs
import llm_detector.functions.helper as helper_funcs
import llm_detector.functions.scoring as scoring_funcs

if __name__ == '__main__':

    # Start logger
    logger=helper_funcs.start_logger()

    # Set-up queues to pass text to and from the scoring loop
    scoring_loop_input_queue=queue.Queue()
    scoring_loop_output_queue=queue.Queue()

    # Initialize Flask app
    flask_app=app_funcs.create_flask_celery_app(scoring_loop_input_queue,scoring_loop_output_queue)
    logger.info('Flask app initialized')

    # Get the Celery app
    celery_app=flask_app.extensions["celery"]
    logger.info('Celery app initialized')

    # Put the Celery into a thread
    celery_app_thread=Thread(
        target=celery_app.worker_main,
        args=[['worker', '--loglevel=INFO']]
    )

    logger.info('Celery app MainProcess thread initialized')

    # Start the Celery app thread
    celery_app_thread.start()
    logger.info('Celery app MainProcess thread started')

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=app_funcs.start,
        args=[flask_app,config.IP_ADDRESS,config.PORT]
    )

    logger.info('Flask app thread initialized')

    # Start the flask app thread
    flask_app_thread.start()
    logger.info('Flask app thread started')

    # Put the main scoring loop in a thread
    scoring_loop_thread=Thread(
        target=scoring_funcs.scoring_loop,
        args=[scoring_loop_input_queue,scoring_loop_output_queue,logger]
    )

    logger.info('Scoring loop thread initialized')

    # Start the flask app thread
    scoring_loop_thread.start()
    logger.info('Scoring loop thread started')


    # # Initialize the flask app
    # app=flask_app.setup(input_queue, output_queue)

    # # Put the flask app into a thread
    # flask_app_thread=Thread(
    #     target=flask_app.start,
    #     args=[app,config.IP_ADDRESS,config.PORT]
    # )

    # # Start the flask app thread
    # flask_app_thread.start()
    # logger.info('Flask app started')

    # # Start main scoring loop
    # while True:

    #     # Check the input queue for a string to score
    #     if input_queue.empty() is False:

    #         # Get the string from the in put queue
    #         suspect_string=input_queue.get()

    #         # Call the scoring function
    #         score=scoring_funcs.score_string(
    #             observer_model,
    #             performer_model,
    #             suspect_string
    #         )

    #         # Send the score and string back to flask
    #         result={
    #             'score': score[0],
    #             'text': suspect_string
    #         }

    #         output_queue.put(result)
