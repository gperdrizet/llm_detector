# Project set-up notes

## Flask: development server

Following the instructions from the Flask documentation's [install guide](https://flask.palletsprojects.com/en/3.0.x/installation/):

### Make virtual environment

```text
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Add *.venv* to *.gitignore* if it's not there already.

### Install flask

```text
pip install flask
pip freeze > requirements.txt
```

### Test it out

Place the following in *app.py*:

```python
'''Simple test of flask development server'''

from flask import Flask # type: ignore

app = Flask(__name__)

@app.route("/")
def hello_world():
    '''Shows simple greeting on homepage'''

    return "<p>Hello, World!</p>"

```

Make sure to allow the default port 5000 through the firewall and start the development server on the LAN, substituting your development box's IP.

```text
sudo ufw allow 5000
flask run --host=192.168.1.148
```

A simple 'hello, World!' landing page should now be visible via a web-browser at <http://192.168.1.148:5000> from any machine on the LAN.
