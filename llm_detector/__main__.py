'''Main module to launch flask app'''

from threading import Thread
import llm_detector.functions.flask_app as flask_app

if __name__ == '__main__':

    app=flask_app.setup()

    flask_app_thread = Thread(target=flask_app.start, args=[app])
    flask_app_thread.start()
