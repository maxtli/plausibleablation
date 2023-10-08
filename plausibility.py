# %%
from model import load_demo_gpt2, tokenizer
from data import retrieve_toxic_data, retrieve_owt_data
# %%

model = load_demo_gpt2()

# %%

toxic_batch_size = 10
owt_batch_size = 10
context_length = 20
# %%

toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
owt_data_loader = retrieve_owt_data(owt_batch_size, context_length, tokenizer)

# %%

def sanity_check(model):
    model(tokens, tokens)

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