'''Throw-away minimal script to test LLaMA3 on local hardware.
Runs one inference from naked prompt on GPU 0. Adapted from 
HuggingFace model card: 

https://huggingface.co/meta-llama/Meta-Llama-3-8B

Emphasis on basic testing rather than quality of output. Will
it generate? Prints raw and untokenized response, along with some 
basic stats.
'''

import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # pylint: disable=import-error

# Manually set device for testing
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']='1'

# Give torch CPU resources
torch.set_num_threads(10)

# The model we want to load
MODEL_ID = "meta-llama/Meta-Llama-3-8B"

# Used 4-bit quantization so the model will fit on one GPU
quantization_config = {
    'load_in_4bit': True,
    #'bnb_4bit_quant_type': 'nf4',
    #'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': torch.float16

}

# Load the model and tokenizer and time it
loading_start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='cuda:0',
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Stop the loading timer
total_load_time = time.time() - loading_start_time

# Set-up to track GPU memory and time
torch.cuda.reset_peak_memory_stats()
generation_start_time = time.time()

# Make and tokenize a message
MESSAGE='Hi! How are you tonight?'

prompt = tokenizer.encode(
    MESSAGE,
    return_tensors = 'pt'
).to('cuda')

# Prompt the model. Set max new tokens to a small value so if the
# model goes off the rails and starts spitting out hallucinations
# We don't have to wait around for it all day.
output_ids = model.generate(
    prompt,
    max_new_tokens=32,
    pad_token_id=tokenizer.eos_token_id
)

reply = tokenizer.batch_decode(output_ids, eos_token_id=tokenizer.eos_token_id)

# Stop the generation timer
total_generation_time = time.time() - generation_start_time

# Get the max memory footprint
max_memory = torch.cuda.max_memory_allocated()

# How many tokens did we generate?
total_tokens = len(output_ids[0])

# How fast did we generate?
generation_rate = round((total_tokens/total_generation_time), 1)

# Print for inspection
print(f'\nRaw output: {output_ids}')
print()
print(f'Un-tokenized reply: {reply}')
print()
print(f'Model loading time: {round(total_load_time, 1)} sec.')
print(f'Tokens generated: {total_tokens}')
print(f'Peak GPU memory use: {round(max_memory / 10**9, 1)} GB')
print(f'Total generation time: {round(total_generation_time, 1)} sec.')
print(f'Generation rate: {round(generation_rate, 1)} tokens per sec.')
print()
