'''Internal LLM detector API.'''

import time
from typing import Callable
from flask import Flask, request # type: ignore

def setup(input_queue: Callable, output_queue: Callable) -> Callable:
    '''Define the flask app'''

    # Initialize flask app
    app=Flask(__name__)

    # Set listener for text strings via POST
    @app.route('/llm_detector', methods=['POST'])
    def echo_text():
        '''Returns user provided string as JSON.'''

        # Get the suspect text string from the request data
        request_data=request.get_json()

        # Put the string in the queue
        input_queue.put(request_data['string'])

        # Wait for the score to be returned in the output queue
        while output_queue.empty():
            time.sleep(1)

        # Get the result once it's ready
        result=output_queue.get()

        return result

    return app

def start(app: Callable, ip_address: str, port: int) -> None:
    '''Starts flask app'''

    app.run(host=ip_address, port=port)
