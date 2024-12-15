# Benchmarking/optimization notebooks outline

## [01-LLaMA3-8B.ipynb](https://github.com/gperdrizet/llm_detector/blob/benchmarking/benchmarking/notebooks/01-LLaMA3-8B.ipynb)

### LLaMA3-8B Benchmarking

1. **Model loading**

    1. Dependent variables: model loading rate, model loading time
    2. Independent variables: source drive, device map, CPU cores

2. **Inference**

    1. Dependent variables: inference rate, inference time
    2. Independent variables: model, device map, quantization, decoding strategy, CPU cores, max new tokens

#### 1. Model loading

##### 1.1. TLDR

##### 1.2. LLaMA3 loading rate

##### 1.3. LLaMA3 loading time averages: 16 CPU cores

##### 1.4. Observations

##### 1.5. Conclusions

#### 2. Inference

##### 2.1. TLDR

##### 2.2. LLaMA3 generation rate: 4-bit vs 8-bit quantization, 4 CPU cores

##### 2.3. LLaMA3 inference time averages: 256 tokens, 4-bit vs 8-bit quantization, 4 CPU cores

##### 2.4. LLaMA3 generation rate: 4-bit quantization

##### 2.5. LLaMA3 inference time averages: 4-bit quantization, 256 tokens & 4 CPU cores

##### 2.6. Observations

##### 2.7. LLaMA3 generation rate: CPU only

##### 2.8. LLaMA3 inference time averages: CPU only, 256 tokens

##### 2.9. Observations

##### 2.10. Inference time averages: all devices, 256 tokens, 4 CPU cores

##### 2.9. Conclusions

#### 3. Decoding strategy

##### 3.1. LLaMA3 generate rate: K80 vs GTX1070

##### 3.2. LLaMA3 inference time averages: K80 vs GTX1070, 256 tokens

##### 3.2. Generation rate by model: GTX1070

##### 3.4. LLaMA3 generation rate, CPU only

##### 3.5. Generation rate by model, CPU only

##### 3.3. Conclusion

## [02-logit_calculation.ipynb](https://github.com/gperdrizet/llm_detector/blob/benchmarking/benchmarking/notebooks/02-logit_calculation.ipynb)

### Logit calculation benchmarking

#### 1. Logits calculation

##### 1.1. TLDR: logits

##### 1.2. Memory footprint, 4-bit quantization

##### 1.3. Logit calculation rate, 4-bit quantization

##### 1.4 Rate and memory averages LLaMA3-8B

##### 1.4. Memory: CPU vs GPU

##### 1.5. Rate: CPU vs GPU

##### 1.6. Conclusions

#### 2. String encoding

##### 2.1. TLDR: encoding

##### 2.2. Plot: memory footprint for device map & input sequence length

##### 2.3. Plot: encoding rate for device map & input sequence length

## [03-perplexity_ratio_score.ipynb](https://github.com/gperdrizet/llm_detector/blob/benchmarking/benchmarking/notebooks/03-perplexity_ratio_score.ipynb)

### Perplexity ratio score exploratory data analysis

#### 1. Hans et al. (2024) Datasets

##### 1.1. Hans datasets structure

##### 1.2. Hans datasets plots

###### 1.2.1. Text length distributions

###### 1.2.2. Human-synthetic text length correlation by record

###### 1.2.3. Text composition distribution over records

###### 1.2.4. TF-IDF distributions

#### 2. Perplexity ratio scores

##### 2.1. Model benchmark

##### 2.2. LLaMA3-8B

#### Perplexity vs cross-perplexity

#### Perplexity ratio score distribution

#### Binoculars score by fragment length

#### Fragment length distribution

#### Thresholding
