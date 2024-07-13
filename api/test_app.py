'''Simple utility to test LLM detector API.'''

import os
import json
import time
import urllib.request

# LLM detector endpoint URL
URL = f"http://{os.environ['HOST_IP']}:{os.environ['FLASK_PORT']}/submit_text"

# Define some test strings
TEST_STRINGS = [
    'This a a sentence written by a human being designed to test the llm detector.',
    'This is another text fragment also written by a human to test the llm detector API.',
    'This is a string to test what happens when the API receives requests in succession.',
    'This a a sentence written by a human being designed to test the llm detector.',
    'This is another text fragment also written by a human to test the llm detector API.',
    'This is a string to test what happens when the API receives requests in succession.'
]

# Loop on the test strings, collecting the result ids
result_ids = []

for test_string in TEST_STRINGS:

    # Assemble the payload
    payload = {'string': test_string}
    json_payload = json.dumps(payload) # Explicitly converts to json
    json_bytes_payload = json_payload.encode('utf-8') # Encodes to bytes

    # Setup the request
    req = urllib.request.Request(URL)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    req.add_header('Content-Length', len(json_bytes_payload))

    # Submit the request
    body = urllib.request.urlopen(req, json_bytes_payload).read()

    # Read and parse the results
    contents = json.loads(body)

    # Collect the result id
    result_ids.append(contents['result_id'])

    print(f'Submitted: {test_string}')
    print(f'Received: {contents}')

# Once all the strings have been submitted, loop on the list of result
# id's checking to see if each one is done, until they all finish
tmp_result_ids=result_ids

print('\nPolling for results:')

while len(result_ids) > 0:
    for result_id in result_ids:

        # Ask for the results from this id
        body = urllib.request.urlopen(
            f'http://192.168.1.148:5000/result/{result_id}').read()

        # Read and parse the results
        contents = json.loads(body)

        if contents['ready'] is True:

            print(f"Author call: {contents['value']['author_call']}")
            tmp_result_ids.remove(result_id)

    result_ids = tmp_result_ids

    # Wait before checking again
    time.sleep(0.1)
