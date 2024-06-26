'''Collection of functions to score strings'''

from typing import Callable
import time
import numpy as np
import torch
import transformers
import llm_detector.classes.llm as llm_class

def scoring_loop(
    scoring_loop_input_queue: Callable,
    scoring_loop_output_queue: Callable,
    logger: Callable
) -> None:

    '''Main loop to score text. Takes text as string from queue and
    returns result to queue.'''

    # Set available CPU cores - doing this from the LLM class does not seem to work
    torch.set_num_threads(16)

    # Configure and load two instances of the model, one base for the observer
    # and one instruct for the performer. Use different GPUs.
    observer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B',
        device_map='cuda:1',
        logger=logger
    )

    observer_model.load()
    logger.info('Loaded observer model')

    performer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B-instruct',
        device_map='cuda:2',
        logger=logger
    )

    performer_model.load()
    logger.info('Loaded performer model')

    # Start main scoring loop
    while True:

        # Check the input queue for a string to score
        if scoring_loop_input_queue.empty() is False:

            # Get the string from the in put queue
            suspect_string=scoring_loop_input_queue.get()

            # Call the scoring function
            score=score_string(
                observer_model,
                performer_model,
                suspect_string
            )

            # Send the score and string back to flask
            result={
                'score': score[0],
                'text': suspect_string
            }

            scoring_loop_output_queue.put(result)

    # Wait before we check the queue again
    time.sleep(1)

def score_string(observer_model: Callable, performer_model: Callable, string: str=None) -> float:
    '''Takes a string, computes and returns llm detector score'''

    # Encode the string using the observer's tokenizer
    encodings=observer_model.tokenizer(
        string,
        return_tensors="pt",
        return_token_type_ids=False
    ).to(observer_model.device_map)

    # Calculate logits
    observer_logits=observer_model.model(**encodings).logits
    performer_logits=performer_model.model(**encodings).logits

    # Calculate perplexity
    ppl=perplexity(encodings, performer_logits)

    # Calculate cross perplexity
    x_ppl=entropy(
        observer_logits.to('cuda:0'),
        performer_logits.to('cuda:0'),
        encodings.to('cuda:0'),
        observer_model.tokenizer.pad_token_id
    )

    binoculars_scores = ppl / x_ppl
    binoculars_scores = binoculars_scores.tolist()

    return binoculars_scores


# Take some care with '.sum(1)).detach().cpu().float().numpy()'. Had errors as cribbed from
# the above repo. Order matters? I don't know, current solution is very 'kitchen sink'

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0
):

    '''Perplexity score function'''

    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.detach().cpu().numpy()

    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0
):

    '''Cross entropy score function'''

    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).detach().cpu().float().numpy())

    return agg_ce
