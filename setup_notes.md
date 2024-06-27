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

## HuggingFace Transformers

Set the *HF_HOME* environment variable via *.venv/bin/activate* to some fast storage:

```text
# .venv/bin/activate
export HF_HOME="/mnt/fast_scratch/huggingface_transformers_cache"
```

Install Transformers and some dependencies:

```text
pip install transformers
pip install torch==1.13.1   # Specific version for NVIDIA K80 with CUDA 11.4
pip install accelerate      # Needed for model quantization
```

Note: According to [the pytorch site](https://pytorch.org/get-started/previous-versions/) 1.13.1 is compatible with CUDA 11.7 and 11.6, but in my testing it's the most recent version which works with my setup - CUDA 11.4 & NVIDIA driver 470 on a pair of Tesla K80s.

## Bitsandbytes

To fit most 7-8 billion parameter sized models (including LLaMA3) we need to quantize with Bitsandbytes - do this from the parent directory. I don't think it actually matters where the source is, as long as we are in the venv when pip installing. For organizational reasons, keep a correctly versioned bitsandbytes around above the project level:

```text
wget https://github.com/TimDettmers/bitsandbytes/archive/refs/tags/0.42.0.tar.gz
tar -xf 0.42.0.tar.gz
cd bitsandbytes-0.42.0
```

Build and install, making sure we have scipy first:

```text
pip install scipy
CUDA_VERSION=117 make cuda11x_nomatmul_kepler
python setup.py install
python -m bitsandbytes
```

Again, CUDA 11.7 sounds like a version mismatch, but empirically, that's what works with out system set-up. Output should be something like:

```text
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/cuda-11.4/targets/x86_64-linux/lib/stubs/libcuda.so
/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudart.so

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++
/mnt/arkk/bitsandbytes-0.42.0/bitsandbytes/bitsandbytes_cuda117_nocublaslt.so
/mnt/arkk/bitsandbytes-0.42.0/build/lib/bitsandbytes/bitsandbytes_cuda117_nocublaslt.so

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++

++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['6.1', '3.7', '3.7']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable


WARNING: Please be sure to sanitize sensible info from any such env vars!

SUCCESS!
Installation was successful!
```

## Step back

OK, doing well so far. At this point the backend works on the flask development server. We can use curl or python's urllib to send in a text string and get back a score. My biggest worry is concurrency/scalability. We are using threads and queues to send strings and scores back and forth between flask and the scoring function, but the way we are doing it blocks the flask app while the scoring function does it's thing. It's basically like we are calling the scoring function directly from the flask route - put a string in the input queue, wait for a score to show up in the output queue, send the result back to the user. For now, this is probably fine - we don't have the compute power to compute scores for more than one user at the same time, but I think we can do better.

After some reading it sounds like the best way to make this production grade is the following:

1. Run flask with Gunicorn, instead of the built-in dev server
2. Use a Celery task queue to send jobs to the GPUs from flask and recive the output
3. This also means we need Redis to act as message broker between the Flask Celery client and the GPU Celery worker

I'm sure there is a hundred ways to do this kind of thing, but I'm hoping that with the above, we can get it set up without too much trouble, have a pathway to scale and most importantly have more confidence about the mapping of string input from client to score output back to client that just FIFO style queue waiting.

Ok, enough rambling for now - all of that means we need to set a few more things up.

## Redis

```text
curl -O http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
rm redis-stable.tar.gz
cd redis-stable
make
```

Then install the Redis python package in the project's virtual environment:

```text
pip install Redis
```

Make finished without issues, but make test gave the following error: *You need tcl 8.5 or newer in order to run the Redis test*.

Not sure how important this is really, but here is the fix:

```text
sudo apt-get install tcl
```

After that, *make test* runs fine. Bind the dev box's IP address on the lan and turn off protected mode to make it available in *redis-stable/redis.conf*. Note: obviously only do this if you are confident that whatever network you are exposing it to is private.

```text
bind 192.168.1.148
protected-mode no
```

Redis defaults to port 6379, os open that and then start the server.

```text
$ sudo ufw allow 6379
$ src/redis-server --protected-mode no

                _._                                                  
           _.-``__ ''-._                                             
      _.-``    `.  `_.  ''-._           Redis 7.2.5 (00000000/0) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._                                  
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 115665
  `-._    `-._  `-./  _.-'    _.-'                                   
 |`-._`-._    `-.__.-'    _.-'_.-'|                                  
 |    `-._`-._        _.-'_.-'    |           https://redis.io       
  `-._    `-._`-.__.-'_.-'    _.-'                                   
 |`-._`-._    `-.__.-'    _.-'_.-'|                                  
 |    `-._`-._        _.-'_.-'    |                                  
  `-._    `-._`-.__.-'_.-'    _.-'                                   
      `-._    `-.__.-'    _.-'                                       
          `-._        _.-'                                           
              `-.__.-'                                               

115665:M 21 Jun 2024 21:45:50.364 * Server initialized
115665:M 21 Jun 2024 21:45:50.365 * Ready to accept connections tcp
```

I'm sure theres a lot more setup we could do and we probably want to put it in a docker container, but let's go with that for now. Leave it in a screen and move on.

## Celery

This on should be pretty easy using the default config:

```text
pip install celery
```

While first playing around with Celery, a depreciation warning was encountered:

```text
PendingDeprecationWarning: The broker_connection_retry configuration setting will no longer determine
whether broker connection retries are made during startup in Celery 6.0 and above.
If you wish to retain the existing behavior for retrying connections on startup,
you should set broker_connection_retry_on_startup to True.
```

Following the instructions given in the warning via the Celery configuration dictionary in the Flask app set-up function clears the warning.

## Next steps

OK, we have the basic API working. One important GOTCHA to note for later. CUDA complains about being reinitialized in Celery workers (fork vs spawn). The solution was to start the Celery worker with *--pool=solo* this makes tasks blocking and causes them to run in-line (ie in the same process as the worker). So, bye-bye concurrency for now. This is OK for the time being, because we don't have the compute resources to compute scores for more than one string at a time, so this behavior is what we want. If we were to scale this, however, the execution and job scheduling strategy would need a revisit.

A few other more pressing items to handle:

1. Run flask with Gunicorn
2. Daemonize/dockerize Redis
3. Decide if we are going to publicly expose the API directly
4. Start working on the interface

If we do want to expose the API directly to the public rather than via a messaging app only, there are a few further things to figure out:

1. Rate limit
2. NGINX configuration
3. Data privacy
4. Documentation

At this point, the README needs a good working over too and we should probably dockerize the whole thing.

## Gunicorn

Gunicorn will be our deployment WSGI server for Flask. Set it up following the Gunicorn instructions in the [Deploying to Production](https://flask.palletsprojects.com/en/3.0.x/deploying/gunicorn/) section of the Flask documentation.
