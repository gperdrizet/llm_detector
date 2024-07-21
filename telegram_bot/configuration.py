'''Module specific constants and configuration.'''

import os

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Other project paths
LOG_PATH = f'{PROJECT_ROOT_PATH}/logs'

# Logging stuff
LOG_LEVEL = 'DEBUG'
LOG_PREFIX='%(levelname)s - %(name)s - %(message)s'
CLEAR_LOGS = True

# Flask app stuff
HOST_IP = os.environ['HOST_IP']
FLASK_PORT = os.environ['FLASK_PORT']

# Bot greeting, set at start of new conversation
BOT_GREETING = "Hi, I'm a bot! I can't talk to you, but if you send me some text, I'll try to determine if it was written by a human or a machine."