# Project set-up notes

## Flask

Following the instructions from the [install guide](https://flask.palletsprojects.com/en/3.0.x/installation/):

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
