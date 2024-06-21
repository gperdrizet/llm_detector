'''Internal LLM detector API.'''

from flask import Flask, request # type: ignore

app = Flask(__name__)

@app.route('/llm_detector', methods=['POST'])
def echo_text():
    '''Returns user provided string as JSON.'''

    # Get and return the suspect text string from the
    # request data
    request_data = request.get_json()
    return {'suspect text': request_data['string']}
