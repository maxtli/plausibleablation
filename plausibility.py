# %%
from transformer import load_demo_gpt2 
from models import import_ablated_model
# %%

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