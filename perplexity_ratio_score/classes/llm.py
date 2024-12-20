'''Class to hold objects and methods related to the LLM'''

from __future__ import annotations

import gc
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import perplexity_ratio_score.configuration as config

class Llm:
    '''LLM class to bundle configuration options, model and tokenizer
    together in once place. Initialize a class instance, override
    default values for loading, quantization or generation, if desired
    then, load model and tokenizer with load method.'''

    def __init__(
        self,
        experiment_name: str=None,
        cache_dir: str=config.CACHE_DIR,
        hf_model_string: str=config.HF_MODEL_STRING,
        model_name: str=config.MODEL_NAME,
        device_map: str=config.DEVICE_MAP,
        quantization: str=config.QUANTIZATION,
        decoding_strategy: str=config.DECODING_STRATEGY,
        bnb_4bit_compute_dtype: str=config.BNB_4BIT_COMPUTE_DTYPE,
        #bnb_8bit_compute_dtype: str=config.BNB_8BIT_COMPUTE_DTYPE,
        max_new_tokens: str=config.MAX_NEW_TOKENS,
        cpu_cores: int=config.CPU_CORES
    ) -> None:

        '''Initial class setup, takes default values from configuration file
        creates placeholders for LLM and tokenizer to be loaded later and
        adds logger'''

        # Add the logger
        self.logger = logging.getLogger(f'{experiment_name}.LLM')

        # Initialize values for loading and generation
        self.cache_dir=cache_dir
        self.hf_model_string=hf_model_string
        self.model_name=model_name
        self.device_map=device_map
        self.quantization=quantization
        self.decoding_strategy=decoding_strategy
        self.bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        #self.bnb_8bit_compute_dtype=bnb_8bit_compute_dtype
        self.cpu_cores=cpu_cores
        self.max_new_tokens=max_new_tokens

        # Reserve loading the tokenizer and model for the load
        # method to give the user a chance to override default
        # parameter values if desired
        self.model=None
        self.tokenizer=None

        # Also add placeholders for quantization and generation
        # configuration, it not modified before model loading
        # model defaults will be used
        self.quantization_config=None
        self.default_generation_config=None
        self.generation_config=None

    def load(self) -> None:
        '''Loads model and tokenizers using whatever configuration 
        values are currently under self'''

        # Set available cores
        torch.set_num_threads(self.cpu_cores)
        self.logger.debug('Torch has %s threads', torch.get_num_threads())
        self.logger.debug('LLM class instance specifies %s', self.cpu_cores)

        # Set quantization configuration with current values under self
        self.set_quantization_config()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_string,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            quantization_config=self.quantization_config
        )

        # Save the freshly loaded model-default generation configuration for later
        self.default_generation_config = GenerationConfig.from_model_config(self.model.config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_string,
            cache_dir=self.cache_dir
        )

    def clear(self) -> None:
        '''Removes model and tokenizer, clears GPU memory'''

        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def prompt(self, message: str):
        '''Generate from input string'''

        # Tokenize the input
        tokenized_message=self.tokenizer.encode(
            message,
            return_tensors='pt'
        )

        # Move tokenized message to GPU, if needed
        if self.device_map != 'cpu':
            tokenized_message=tokenized_message.to('cuda')

        # Set the generation configuration based on current decoding strategy
        self.set_generation_config()

        # Log generation configuration for debug
        self.logger.debug(self.generation_config)

        # Prompt the model
        try:
            output_ids = self.model.generate(
                tokenized_message,
                generation_config=self.generation_config
            )

            # Un-tokenize the response
            reply = self.tokenizer.batch_decode(
                output_ids,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Catch value errors - seem to arise from conflicting generation
        # configuration parameter settings sometimes
        except ValueError as value_error:

            # Log error for debug
            self.logger.error(value_error)

            # Set return values
            reply=None
            output_ids=None

        # Log reply for debug
        self.logger.debug('Reply: %s', reply)

        return reply, output_ids

    def set_quantization_config(self) -> None:
        '''Builds quantization configuration dictionary'''

        # Construct quantization configuration dictionary based
        # on requested quantization type
        if self.quantization == '4-bit':
            self.quantization_config={
                'load_in_4bit': True,
                'load_in_8bit': False,
                'bnb_4bit_compute_dtype': self.bnb_4bit_compute_dtype
            }

        elif self.quantization == '8-bit':
            self.quantization_config={
                'load_in_4bit': False,
                'load_in_8bit': True,
                'bnb_4bit_compute_dtype': self.bnb_4bit_compute_dtype
            }

        # If we are setting device map to CPU, override quantization
        if self.device_map == 'cpu':
            self.quantization='None'
            self.quantization_config=None

    def set_generation_config(self) -> None:
        '''Builds generation configuration'''

        # Make a copy of the model-default generation configuration
        # and update the desired parameters based on the requested
        # decoding strategy
        self.generation_config=self.default_generation_config

        if self.decoding_strategy == 'Greedy':
            self.generation_config.do_sample=False
            self.generation_config.num_beams=1
            self.generation_config.top_p=1
            self.generation_config.temperature=1

        elif self.decoding_strategy == 'Contrastive':
            self.generation_config.do_sample=False
            self.generation_config.num_beams=1
            self.generation_config.top_p=1
            self.generation_config.temperature=1
            self.generation_config.penalty_alpha=0.6
            self.generation_config.top_k=4

        elif self.decoding_strategy == 'Multinomial sampling':
            self.generation_config.do_sample=True
            self.generation_config.num_beams=1
            self.generation_config.top_p=1
            self.generation_config.temperature=1

        elif self.decoding_strategy == 'Beam-search':
            self.generation_config.do_sample=False
            self.generation_config.num_beams=5
            self.generation_config.top_p=1
            self.generation_config.temperature=1

        elif self.decoding_strategy == 'Beam-search sampling':
            self.generation_config.do_sample=True
            self.generation_config.num_beams=5
            self.generation_config.top_p=1
            self.generation_config.temperature=1

        elif self.decoding_strategy == 'Diverse beam-search':
            self.generation_config.do_sample=False
            self.generation_config.num_beams=5
            self.generation_config.top_p=1
            self.generation_config.temperature=1
            self.generation_config.num_beam_groups=5
            self.generation_config.diversity_penalty=1.0

        elif self.decoding_strategy == 'Model default':
            pass

        # Add some parameters that don't depended decoding strategy
        self.generation_config.max_new_tokens=self.max_new_tokens
        self.generation_config.pad_token_id=self.tokenizer.eos_token_id
