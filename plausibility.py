# %%
from model import load_demo_gpt2, tokenizer
from data import retrieve_toxic_data, retrieve_owt_data
import torch
from torch import linalg as LA
# %%

model = load_demo_gpt2()

# %%

toxic_batch_size = 10
owt_batch_size = 10
context_length = 20
# %%

# importantly, the two should have the same context length
toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
owt_data_loader = retrieve_owt_data(owt_batch_size, context_length, tokenizer)

# %%

for batch in toxic_data_loader:
    owt_batch = next(owt_data_loader)['tokens']

    # remove start token
    batch = batch[:,1:].long()
    owt_batch = owt_batch[:, 1:].long()

    # shape: batch n_heads (0 alt, 1 original) seq_len vocab_size
    logits = model(batch, owt_batch)[0]  # 0 is the logits
    
    probs = torch.softmax(logits, dim=-1)

    orig_probs = probs[:,1]
    alt_probs = probs[:,0]
    plausible_probs = probs[:,2:]

    orig_gradients = plausible_probs / orig_probs.unsqueeze(1)
    orig_gradients -= torch.mean(orig_gradients, dim=-1).unsqueeze(-1)
    orig_norms = LA.norm(orig_gradients, dim=-1)

    alt_gradients = plausible_probs / alt_probs.unsqueeze(1)
    alt_gradients -= torch.mean(alt_gradients, dim=-1).unsqueeze(-1)
    alt_norms = LA.norm(alt_gradients, dim=-1)

    c_weight = LA.norm(orig_norms * alt_gradients + alt_norms * orig_gradients, dim=-1) / orig_norms * alt_norms

    plausibility_scores = (1 - c_weight) * orig_norms/ (orig_norms + alt_norms)
    

    
    
    # sum of loss over the entire sequence


def infer_batch(model, criterion, batch, batch_size, demos, device="cuda"):

    # cast the entire batch tensor to torch.long
    batch = batch.long()

    # remove start token 
    batch = batch[:, 1:]
    
    # concatenate the demos and the batch
    # if batch size is < batch_size, remove some demos
    if batch.shape[0] < batch_size:
        demos = demos[:batch.shape[0]]
    input = torch.cat([demos, batch], dim=1)

    # generate the output
    out = model(input)[0]  # 0 is the logits

    return evaluate_sequence_loss(out, input, criterion, demos.shape[1])


# logits shape: n_heads seq_len vocab_size
def sanity_check(model):
    for batch in toxic_data_loader:
        logits = model(batch, batch)

# %%

# dataloader

# perform inference with batched toxic samples 
# perform inference with untoxic samples
# perform inference with ablated untoxic samples
# take specific untoxic examples from the finetuned model, and perform inference
# do this 144x, once for each attention head. do i need to save the indices? (also, ???)
# (i guess this is just activation patching)

# do some arithmetic on the output logits
# check the ablated loss on the toxic samples

# 