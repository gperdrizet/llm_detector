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
