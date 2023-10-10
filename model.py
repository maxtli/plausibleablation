# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import DemoTransformer, Config
from transformer_lens import HookedTransformer
import torch
import pickle

# %%

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
RATIO = 0.2

# %% 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
ref_model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
ref_model.to(DEVICE)

def load_demo_gpt2():
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False, pad_token_id=tokenizer.eos_token_id)
    demo_gpt2 = DemoTransformer(Config(debug=False))
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    demo_gpt2.to(DEVICE)
    return demo_gpt2

