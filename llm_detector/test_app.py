'''Simple utility to test LLM detector API.'''

import json
import urllib.request

# LLM detector endpoint URL
URL='http://192.168.1.148:5000/submit_text'

# Define some test strings
TEST_STRINGS=[
    'This a a sentence written by a human being designed to test the llm detector.',
    'This is another text fragment also written by a human to test the llm detector API',
    'This is a string to test what happens when the API receives requests in succession',
    'This a a sentence written by a human being designed to test the llm detector.',
    'This is another text fragment also written by a human to test the llm detector API',
    'This is a string to test what happens when the API receives requests in succession'
]

# Loop on the test strings
for test_string in TEST_STRINGS:

    # Assemble the payload
    payload={'string': test_string}
    json_payload=json.dumps(payload) # Explicitly converts to json
    json_bytes_payload=json_payload.encode('utf-8') # Encodes to bytes

    # Setup the request
    req = urllib.request.Request(URL)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    req.add_header('Content-Length', len(json_bytes_payload))

    # Submit the request
    body=urllib.request.urlopen(req, json_bytes_payload).read()
    contents=json.loads(body)

    # Print the response
    print(contents)
