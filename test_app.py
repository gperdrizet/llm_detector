'''Simple utility to test LLM detector API.'''

import json
import urllib.request

# LLM detector endpoint URL
URL='http://192.168.1.148:5000/llm_detector/'

# Define and encode the test string
TEST_STRING='this is the test string'
safe_test_string=urllib.parse.quote(TEST_STRING)

# Assemble the payload
payload=f'{URL}{safe_test_string}'

# Make the request
body=urllib.request.urlopen(payload).read()
contents=json.loads(body)

print(f'Reply is: {type(contents)}')
print(contents)
