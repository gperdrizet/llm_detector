# LLaMA3 Setup Notes

Let's get LLaMA3 set-up via HuggingFace.

## Environment

First make a new venv:

```text
python3 -m venv ./.venv
```

Then, set the *HF_HOME* environment variable via *.venv/bin/activate* to some fast storage:

```text
# .venv/bin/activate
export HF_HOME="/mnt/fast_scratch/huggingface_transformers_cache"
```

Now, activate, update pip and install some basic dependencies:

```text
source .venv/bin/activate
pip install --upgrade pip
pip install transformers
pip install torch==1.13.1   # Specific version for NVIDIA K80 with CUDA 11.4
pip install accelerate      # Needed for model quantization
```

Note: According to [the pytorch site](https://pytorch.org/get-started/previous-versions/) 1.13.1 is compatible with CUDA 11.7 and 11.6, but in my testing it's the most recent version which works with my setup - CUDA 11.4 & NVIDIA driver 470 on a pair of Tesla K80s.

### Model quantization

Early attempts with the test code from the [HuggingFace model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) went OOM when run on a single K80 chip. Rats. Let's try quantizing it. First need bitsandbytes - do this from the parent directory. I don't think it actually matters where the source is, as long as we are in the venv when pip installing. For organizational reasons, keep a correctly versioned bitsandbytes around above the project level:

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

Now test with *llama_test.py*. First run may take a minute or two as the model and tokenizer is downloaded from HuggingFace, but then we should see the raw token and decoded output and some basic run stats.

```text
$ cd ../LLaMA3
$ python ./llama_test.py

Raw output: tensor([[128000,  13347,      0,   2650,    527,    499,  18396,     30,    358,
           1097,   1695,     11,    358,   1120,   8220,    856,    502,   2363,
             11,    330,    791,   8769,  19558,      1,    555,  43833,  83984,
            942,  99994,     13,   1102,    574,   1633,   1695,      0,    358,
          15262,    433,    264,   2763]], device='cuda:0')

Un-tokenized reply: ['<|begin_of_text|>Hi! How are you tonight? I am good, I just finished my new book, "The Secret Garden" by Frances Hodgson Burnett. It was very good! I liked it a lot']

Model loading time: 13.9 sec.
Tokens generated: 40
Peak GPU memory use: 6.2 GB
Total generation time: 5.9 sec.
Generation rate: 6.7 tokens per sec.
```

Ok, good - we can at least lift the model with 4 bit quantization and device map on 'auto'. Takes about 6.5 GB of GPU memory. Lot's of things to try here - we can mess with the device map, try different loading parameters. I even might like to try putting the model on a RAM disk to save time loading it while we are developing and testing. Not to mention working on an actual prompt. I think the best ting to do before we actually launch into the real project is to write a simple test harness and play with some of this stuff and see how fast/efficient we can make it.

### Other useful modules

### Jupyter notebooks

I like to do plotting an analysis in Jupyter notebooks, so let's set that up for VSCode now. First, install jupyter & matplotlib in the venv:

```text
pip install ipykernel
pip install matplotlib
```

Now we can create an new notebook using the VSCode command pallet (ctrl+shift+p). Make sure to set the python interpreter via VSCode (ctrl+shift+p -> *Python: Select Interpreter*) and then select the same for the kernel in the Jupyter notebook (ctrl+shit+p -> *Notebook: Select Notebook Kernel*). Every once in a while, this seems to stop working. Fix is usually to nuke .vscode-server-insiders directory in the server's home directory, then update the local VSCode install.

### Ramdisk model cache

Let's set-up a basic RAM disk to cache the model(s) we are working with. My hunch is that we can save ourselves some time in startup, especially since as we are developing, testing and troubleshooting we will probably be starting and restarting over and over. Quick look on the fast scratch disk says the vanilla LLaMA3 is 15 GB on disk. Let's go with 32 GB:

```text
$ sudo mkdir /mnt/ramdisk
$ sudo mount -o size=32G -t tmpfs none /mnt/ramdisk
$ sudo chown siderealyear:siderealyear /mnt/ramdisk
$ mkdir /mnt/ramdisk/huggingface_transformers_cache
$ df -h

Filesystem                                 Size  Used Avail Use% Mounted on
udev                                        63G     0   63G   0% /dev
tmpfs                                       13G  1.9M   13G   1% /run
/dev/sda2                                  440G   67G  351G  16% /
tmpfs                                       63G     0   63G   0% /dev/shm
tmpfs                                      5.0M     0  5.0M   0% /run/lock
tmpfs                                       63G     0   63G   0% /sys/fs/cgroup
/dev/nvme0n1                               916G  402G  468G  47% /mnt/fast_scratch
192.168.1.123:/home/siderealyear/ark       3.6T  1.6T  1.9T  45% /mnt/ark
192.168.2.1:/mnt/arkk                       15T  8.6T  6.1T  59% /mnt/arkk
192.168.1.123:/home/siderealyear/big_itch  3.6T  2.4T  1.1T  69% /mnt/big_itch
tmpfs                                       13G     0   13G   0% /run/user/1000
none                                        32G     0   32G   0% /mnt/ramdisk
```

OK, cool. Now we just need that test harness so we can load the models a few times from the fast_scratch cache and the ramdisk to see if it actually saves us any time.
