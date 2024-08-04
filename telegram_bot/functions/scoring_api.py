'''Functions to interact with scoring API'''

import json
import time
import urllib.request
import telegram_bot.configuration as config


async def submit_text(
        suspect_text: str = None,
        response_mode: str = 'default'
) -> str:

    '''Sends user's suspect text to scoring api, get's back a result id
    so we can poll and wait for the scoring backend to do it's thing.'''

    # Assemble the payload
    payload = {'string': suspect_text, 'response_mode': response_mode}
    json_payload = json.dumps(payload) # Explicitly converts to json
    json_bytes_payload = json_payload.encode('utf-8') # Encodes to bytes

    # Setup the request
    req = urllib.request.Request(f'''http://{config.HOST_IP}:\
                                 {config.FLASK_PORT}/submit_text''')
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    req.add_header('Content-Length', len(json_bytes_payload))

    # Submit the request
    body = urllib.request.urlopen(req, json_bytes_payload).read()

    # Read and parse the results
    contents = json.loads(body)

    # Collect the result id
    result_id = contents['result_id']

    return result_id


async def retrieve_result(result_id: str = None) -> str:
    '''Polls for result id, returns result content'''

    while True:

        # Ask for the results from this id
        body = urllib.request.urlopen(
            f'''http://{config.HOST_IP}:\
                {config.FLASK_PORT}/result/{result_id}''').read()

        # Read and parse the results
        contents = json.loads(body)

        if contents['ready'] is True:

            reply = contents['value']['reply']
            return reply

        # Wait before checking again
        time.sleep(0.1)
