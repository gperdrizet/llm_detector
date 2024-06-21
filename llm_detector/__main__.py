import llm_detector.functions.flask_app as flask_app

if __name__ == '__main__':

    app=flask_app.setup()
    app.run(host='192.168.1.148', port=5000)
