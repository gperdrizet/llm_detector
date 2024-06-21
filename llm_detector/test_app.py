'''Simple utility to test LLM detector API.'''

import json
import urllib.request

# LLM detector endpoint URL
URL='http://192.168.1.148:5000/llm_detector'

# Define and encode the test string
TEST_STRING='this is the test string with some special symbols: +, \n, \''

# Assemble the payload
payload={'string': TEST_STRING}
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
print(f'Reply is: {type(contents)}')
print(contents)
