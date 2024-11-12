'''Collection of functions to score strings'''

from typing import Callable
import numpy as np
import torch
import transformers
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import functions.helper as helper_funcs
import configuration as config

# Download nltk assets
download('stopwords')
download('wordnet')

def score_string(
        reader_model: Callable,
        writer_model: Callable,
        perplexity_ratio_kld_kde: Callable,
        tfidf_luts: Callable,
        tfidf_kld_kde: Callable,
        model: Callable,
        string: str=None,
        response_mode: str='default'
) -> float:

    '''Takes a string, computes and returns llm detector score'''

    # To run the XGBoost classifier, we need the following 9 features for
    # this fragment:
    feature_names=[
        'Fragment length (tokens)',
        'Perplexity',
        'Cross-perplexity',
        'Perplexity ratio score',
        'Perplexity ratio Kullback-Leibler score',
        'Human TF-IDF',
        'Synthetic TF-IDF',
        'TF-IDF score',
        'TF-IDF Kullback-Leibler score'
    ]

    # Empty holder for features
    features=[]

    ###############################################################
    # Get perplexity, cross-perplexity and perplexity ratio score #
    ###############################################################

    with torch.no_grad():

        # Encode the string using the reader's tokenizer
        encodings=reader_model.tokenizer(
            string,
            return_tensors='pt',
            return_token_type_ids=False
        ).to(reader_model.device_map)

        # Get the string length in tokens and add to features
        fragment_length=encodings['input_ids'].shape[1]
        features.append(fragment_length)

        # Calculate logits
        reader_logits=reader_model.model(**encodings).logits
        writer_logits=writer_model.model(**encodings).logits

    # Calculate perplexity and add to features
    ppl=perplexity(encodings, writer_logits)
    features.append(ppl[0])

    # Calculate cross perplexity and add to features
    x_ppl=entropy(
        reader_logits.to(config.CALCULATION_DEVICE),
        writer_logits.to(config.CALCULATION_DEVICE),
        encodings.to(config.CALCULATION_DEVICE),
        reader_model.tokenizer.pad_token_id
    )

    features.append(x_ppl[0])

    # Calculate perplexity ratio and add to features
    scores=ppl / x_ppl
    scores=scores.tolist()
    perplexity_ratio_score=scores[0]
    features.append(perplexity_ratio_score)

    ###############################################################
    # Get perplexity ratio Kullback-Leibler score #################
    ###############################################################

    # Calculate perplexity ratio KLD score and add to features
    perplexity_ratio_kld_score=perplexity_ratio_kld_kde.pdf(perplexity_ratio_score)
    features.append(perplexity_ratio_kld_score[0])

    ###############################################################
    # Get human and synthetic TF-IDFs and TF-IDF score ############
    ###############################################################

    # Clean the test for TF-IDF scoring
    sw=stopwords.words('english')
    lemmatizer=WordNetLemmatizer()

    cleaned_string=helper_funcs.clean_text(
        text=string,
        sw=sw,
        lemmatizer=lemmatizer
    )

    # Split cleaned string into words
    words=cleaned_string.split(' ')

    # Initialize TF-IDF sums
    human_tfidf_sum=0
    synthetic_tfidf_sum=0

    # Score the words using the human and synthetic luts
    for word in words:

        if word in tfidf_luts['human'].keys():
            human_tfidf_sum += tfidf_luts['human'][word]

        if word in tfidf_luts['synthetic'].keys():
            synthetic_tfidf_sum += tfidf_luts['synthetic'][word]

    # Get the means and add to features
    human_tfidf_mean=human_tfidf_sum / len(words)
    synthetic_tfidf_mean=synthetic_tfidf_sum / len(words)
    dmean_tfidf=human_tfidf_mean - synthetic_tfidf_mean
    product_normalized_dmean_tfidf=dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

    features.append(human_tfidf_mean)
    features.append(synthetic_tfidf_mean)
    features.append(product_normalized_dmean_tfidf)

    ###############################################################
    # Get TF-IDF Kullback-Leibler score ###########################
    ###############################################################

    # Calculate TF-IDF LKD score and add to features
    tfidf_kld_score=tfidf_kld_kde.pdf(product_normalized_dmean_tfidf)
    features.append(tfidf_kld_score[0])

    ###############################################################
    # Run inference with the classifier ############################
    ###############################################################

    # Make prediction
    prediction=model.predict_proba([features])[0]

    if response_mode == 'default':

        return prediction

    elif response_mode == 'verbose':

        return [prediction[0], prediction[1], dict(zip(feature_names, features))]

# Take some care with '.sum(1)).detach().cpu().float().numpy()'. Had
# errors as cribbed from the above repo. Order matters? I don't know,
# current solution is very 'kitchen sink'

ce_loss_fn=torch.nn.CrossEntropyLoss(reduction = 'none')
softmax_fn=torch.nn.Softmax(dim = -1)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool=False,
    temperature: float=1.0
):

    '''Perplexity score function'''

    shifted_logits=logits[..., :-1, :].contiguous() / temperature
    shifted_labels=encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask=encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan=(ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl=np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl=(ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl=ppl.detach().cpu().numpy()

    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool=False,
    sample_p: bool=False,
    temperature: float=1.0
):

    '''Cross entropy score function'''

    vocab_size=p_logits.shape[-1]
    total_tokens_available=q_logits.shape[-2]
    p_scores, q_scores=p_logits / temperature, q_logits / temperature

    p_proba=softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba=torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores=q_scores.view(-1, vocab_size)

    ce=ce_loss_fn(
        input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask=(encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan=ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce=np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce=(((ce * padding_mask).sum(1) / padding_mask.sum(1)
                   ).detach().cpu().float().numpy())

    return agg_ce
