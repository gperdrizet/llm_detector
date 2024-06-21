'''Main module to launch flask app'''

from threading import Thread
import llm_detector.configuration as config
import llm_detector.functions.flask_app as flask_app

if __name__ == '__main__':

    # Initialize the flask app
    app=flask_app.setup()

    # Put the flask app into a thread
    flask_app_thread=Thread(
        target=flask_app.start,
        args=[app,config.IP_ADDRESS,config.PORT]
    )

    # Start the flask app thread
    flask_app_thread.start()
