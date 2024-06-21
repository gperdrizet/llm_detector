'''Internal LLM detector API.'''

from typing import Callable
from flask import Flask, request # type: ignore

def setup() -> Callable:
    '''Define the flask app'''

    # Initialize flask app
    app=Flask(__name__)

    # Set listener for text strings via POST
    @app.route('/llm_detector', methods=['POST'])
    def echo_text():
        '''Returns user provided string as JSON.'''

        # Get and return the suspect text string from the
        # request data
        request_data = request.get_json()
        return {'suspect text': request_data['string']}

    return app

def start(app: Callable, ip_address: str, port: int) -> None:
    '''Starts flask app'''

    app.run(host=ip_address, port=port)
