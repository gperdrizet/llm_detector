'''Main module to launch flask app and scoring backend'''

import queue
from threading import Thread
import torch
import llm_detector.classes.llm as llm_class
import llm_detector.configuration as config
import llm_detector.functions.flask_app as flask_app
import llm_detector.functions.helper as helper_funcs
import llm_detector.functions.scoring as scoring_funcs

if __name__ == '__main__':

    # Start logger
    logger=helper_funcs.start_logger()
    logger.info('Starting LLM detector')

    # Set-up queues to pass string from flask
    # to the scoring loop and the result back
    # from the scoring loop to flask
    input_queue=queue.Queue()
    output_queue=queue.Queue()

    # Initialize the flask app
    app=flask_app.setup(input_queue, output_queue)

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=flask_app.start,
        args=[app,config.IP_ADDRESS,config.PORT]
    )

    # Start the flask app thread
    flask_app_thread.start()
    logger.info('Flask app started')

    # Set available CPU cores - doing this from the LLM class does not seem to work
    torch.set_num_threads(16)

    # Instantiate two instances of the model, one base for the observer
    # and one instruct for the performer. Use different GPUs.
    observer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B',
        device_map='cuda:1',
        logger=logger
    )

    performer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B-instruct',
        device_map='cuda:2',
        logger=logger
    )

    # Load the models
    observer_model.load()
    performer_model.load()

    # Set the models to evaluation mode to deactivate any dropout modules
    # the is done to ensure reproducibility of results during evaluation
    observer_model.model.eval()
    performer_model.model.eval()

    # Add end of sequence for the pad token if one has not been defined
    if not observer_model.tokenizer.pad_token:
        observer_model.tokenizer.pad_token=observer_model.tokenizer.eos_token

    # Start main scoring loop
    while True:

        # Check the input queue for a string to score
        if input_queue.empty() is False:

            # Get the string from the in put queue
            suspect_string=input_queue.get()

            # Call the scoring function
            score=scoring_funcs.score_string(
                observer_model,
                performer_model,
                suspect_string
            )

            # Send the score back to flask
            output_queue.put(score[0])
