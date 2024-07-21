'''Functions to interact with scoring API'''

import json
import asyncio
import time
import urllib.request
import telegram_bot.configuration as config

async def submit_text(suspect_text: str = None) -> str:
    '''Sends user's suspect text to scoring api, '''

    # Assemble the payload
    payload = {'string': suspect_text}
    json_payload = json.dumps(payload) # Explicitly converts to json
    json_bytes_payload = json_payload.encode('utf-8') # Encodes to bytes

    # Setup the request
    req = urllib.request.Request(f'http://{config.HOST_IP}:{config.FLASK_PORT}/submit_text')
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    req.add_header('Content-Length', len(json_bytes_payload))

    # Submit the request
    body = urllib.request.urlopen(req, json_bytes_payload).read()

    # Read and parse the results
    contents = json.loads(body)

    # Collect the result id
    result_id = contents['result_id']

    return result_id


async def retreive_result(result_id: str = None) -> str:
    '''Polls for result id, returns result content'''

    while True:

        # Ask for the results from this id
        body = urllib.request.urlopen(
            f'http://{config.HOST_IP}:{config.FLASK_PORT}/result/{result_id}').read()

        # Read and parse the results
        contents = json.loads(body)

        if contents['ready'] is True:

            reply = contents['value']['author_call']
            return reply

        # Wait before checking again
        time.sleep(0.1)