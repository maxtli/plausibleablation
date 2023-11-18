# %%

import datasets
import re
import torch
from itertools import cycle
from data import retrieve_owt_data
from model import tokenizer, load_demo_gpt2
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from plausibility_utils import save_files

# %%

device = "cuda:0" if torch.cuda.is_available else "cpu"

# %%
# ioi_ds = datasets.load_dataset("fahamu/ioi")
# ioi_ds.save_to_disk("data/ioi/ioi")

# %%

ioi_ds = datasets.load_from_disk("data/ioi/ioi")

# new_ds = ioi_ds['train'].select(range(100)).map(lambda x: {"io": x['ioi_sentences'].split(' ')[-1], "s": re.split(r'. |, |[Aa]fterwards ',x['ioi_sentences'])[-1].split(' ')[0]})
# %%
model = load_demo_gpt2()
model.to(device).eval()

# %%


# %%

batch_size = 8
max_context_length = 35
# %%

data_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)

owt_loader = retrieve_owt_data(batch_size, max_context_length, tokenizer)
kl_loss = torch.nn.KLDivLoss()
owt_iter = cycle(owt_loader)

# %%

def kldiv_loss(guess, truth):
    return torch.sum(truth * (truth.log() - guess.log()), dim=-1)

def retrieve_plausibility_scores(batch, logits, find_last_token=True):
    # logits shape: batch, head, seq_len, token
    probs = torch.softmax(logits, dim=-1)

    alt_estimate_losses = []
    orig_ood_approxes = []
    head_ood_approxes = []
    

    if find_last_token:
        # full sequence includes the IO
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1
    else:
        last_token_pos = -1 * torch.ones(batch.shape[0]).to(device)

    for id_in_batch in range(probs.shape[0]):
        l_probs = probs[id_in_batch,:,last_token_pos[id_in_batch]]

        orig_probs = l_probs[1]
        alt_probs = l_probs[0]

        alt_estimate_losses.append(kldiv_loss(alt_probs, orig_probs))
        orig_ood_approxes.append(kldiv_loss(orig_probs, alt_probs))

        # head_ood_approxes.append(kldiv_loss(l_probs[2:],alt_probs.unsqueeze(-1)))

        # print(head_ood_approxes[0].shape)

        head_ood_approx = []
        
        for head in range(2, l_probs.shape[0]):
            plausible_probs = l_probs[head]
            head_ood_approx.append(kldiv_loss(plausible_probs, alt_probs))

        head_ood_approxes.append(torch.stack(head_ood_approx,dim=0))
    
    alt_estimate_losses = torch.stack(alt_estimate_losses,dim=0)
    orig_ood_approxes = torch.stack(orig_ood_approxes,dim=0)
    head_ood_approxes = torch.stack(head_ood_approxes,dim=0)

    return alt_estimate_losses, orig_ood_approxes, head_ood_approxes

# %%
alt_orig_ests = []
orig_alt_ests = []
patched_alt_ests = []
for i,b in tqdm(enumerate(data_loader)):
    batch = tokenizer(b['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
    owt_batch = next(owt_iter)['tokens']
    owt_batch = owt_batch[torch.randperm(owt_batch.shape[0]), :batch.shape[1]].to(device)

    with torch.no_grad():
        logits = model(batch, owt_batch)
    
    alt_orig_est, orig_alt_est, patched_alt_est = retrieve_plausibility_scores(batch, logits)

    alt_orig_ests.append(alt_orig_est)
    orig_alt_ests.append(orig_alt_est)
    patched_alt_ests.append(patched_alt_est)

    record_every = 100
    start_idx=0
    if i % record_every == record_every - 1:
        idx = int(i // record_every + start_idx)
        save_files({"alt_orig_ests": alt_orig_ests, "orig_alt_ests": orig_alt_ests, "patched_alt_ests": patched_alt_ests}, path="ioi/simple/", i=idx, cat=True)
        alt_orig_ests = []
        orig_alt_ests = []
        patched_alt_ests = []

    # print(batch)
    # # print(tokenizer(b['ioi_sentences']))
    # print(b)
    # break


# %%
