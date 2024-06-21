'''Internal LLM detector API.'''

from flask import Flask, request # type: ignore

app = Flask(__name__)

@app.get('/llm_detector/<string>')
def echo_text(string):
    '''Returns user provided string as JSON.'''

    return {'suspect text': string}
