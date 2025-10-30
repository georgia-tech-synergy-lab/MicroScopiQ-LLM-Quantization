import time

import torch
import torch.nn as nn
import sys
sys.path.append("../")
from utils.quant_model import quantize_model
from number_system.mx import finalize_mx_specs
from transformers import AutoTokenizer
import tqdm
DEV = torch.device('cuda:0')

def compare_models_and_check_nans(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        # Ensure names are identical
        assert name1 == name2, f"Parameter names differ: {name1} vs {name2}"
        # Check for any NaN values
        if torch.isnan(param1).any() or torch.isnan(param2).any():
            return False, f"NaN found in parameters: {name1}"
        # Ensure parameters are the same
        if not torch.allclose(param1, param2, atol=1e-6):  # You can adjust the tolerance level
            return False, f"Parameters differ: {name1}"
    return True, "Models are the same and no NaNs detected"


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_infer(model, testenc, dev):
    print('Evaluating ...')

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    input_text = "Write a song"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(dev)
    model = model.to(dev)
    output_ids = model.generate(input_ids)
    print(tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True))

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    model.to(dev)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []

    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to(dev)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

if __name__ == '__main__':
    import argparse
    from utils.data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.')

    args = parser.parse_args()

    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'fp4',
        'a_elem_format': 'int4',
        'block_size': 128,
        'custom_cuda': False,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
    }

    mx_specs = finalize_mx_specs(mx_specs)

    model = get_opt(args.model)
    model.eval()
    # print(model)
    # sys.exit()
    q_model = quantize_model(model, mx_specs)
    print(q_model)

    are_same, message = compare_models_and_check_nans(model, q_model)
    if not are_same:
        print(message)  # Prints error message if there's a problem
        sys.exit("Stopping execution due to model mismatch or NaNs in parameters")


    datasets = ['wikitext2', 'ptb', 'c4'] 

    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        print ("Evaluating")
        opt_eval(q_model, testloader, DEV)
        break