# %%
from model import load_demo_gpt2, tokenizer, DEVICE
from data import retrieve_toxic_data, retrieve_owt_data
import torch
import numpy as np
from torch import linalg as LA
from tqdm import tqdm
from itertools import cycle
import seaborn as sns
import pandas as pd
import pickle
# %%

model = load_demo_gpt2()
model.eval()

# %%

toxic_batch_size = 10
owt_batch_size = 10
context_length = 20
# %%

# if toxic data loader and owt data loader are relatively small numbers, then it works better if they are relatively prime.

toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
owt_data_loader = retrieve_owt_data(owt_batch_size, context_length, tokenizer)
# importantly, the two should have the same context length
kl_loss = torch.nn.KLDivLoss()
owt_iter = cycle(owt_data_loader)

# %%
# # %%
# import pickle 
# als = torch.cat(all_losses, dim=0).mean(dim=0).cpu().numpy()
# with open("outputs/zablation_losses.pkl", "wb") as f:
#     pickle.dump(als, f)

# %%
def run_plausibility_scoring():
    files_saved = 0
    all_plausibility_scores = None
    all_c_weights = None
    all_losses = None
    all_resample_loss = None
    all_alt_resample_loss = None
    all_kldiv_indx = []
    all_kldiv_losses = []
    all_kldiv_alt_losses = []
    all_kldiv_comp_losses = []

    for epoch in tqdm(range(1000)):
        for c, batch in enumerate(toxic_data_loader):
            # print("Batches complete:", batches)
            # batches += 1

            if batch.shape[0] < toxic_batch_size:
                continue

            owt_batch = next(owt_iter)['tokens']
            while owt_batch.shape[0] < batch.shape[0]:
                owt_batch = next(owt_iter)['tokens']

            # remove start token
            # randomize the OWT batch
            batch = batch[:,1:].long().to(DEVICE)
            owt_batch = owt_batch[torch.randperm(owt_batch.shape[0]), 1:].long().to(DEVICE)

            # shape: batch n_heads (0 alt, 1 original) seq_len vocab_size
            with torch.no_grad():
                logits = model(batch, owt_batch)
            
            probs = torch.softmax(logits, dim=-1)

            batch_plausibility_scores = None
            batch_c_weight = None
            batch_resample_loss = None
            batch_alt_resample_loss = None
            batch_loss = []
            
            # because batching requires too much memory
            for id_in_batch in range(probs.shape[0]):
                orig_probs = probs[id_in_batch,1]
                alt_probs = probs[id_in_batch,0]
                
                # seq_len
                kl_div = torch.sum(orig_probs * (orig_probs.log() - alt_probs.log()), dim=-1)

                head_plausibility_scores = []
                head_c_weight = []
                head_resample_loss = []
                head_alt_resample_loss = []

                for head in range(2, probs.shape[1]):
                    plausible_probs = probs[id_in_batch,head]

                    # orig_gradients = orig_probs / plausible_probs
                    # orig_gradients -= torch.mean(orig_gradients, dim=-1).unsqueeze(-1)
                    orig_gradients = orig_probs - plausible_probs

                    # seq_len
                    orig_norms = LA.norm(orig_gradients, dim=-1)

                    # alt_gradients = alt_probs / plausible_probs
                    # alt_gradients -= torch.mean(alt_gradients, dim=-1).unsqueeze(-1)
                    alt_gradients = alt_probs - plausible_probs

                    # if original gradients are infinite, and alt gradients are not zero, then c=1
                    # if original gradients are?
                    # if (torch.isnan(orig_gradients) + torch.isnan()).

                    # seq_len
                    alt_norms = LA.norm(alt_gradients, dim=-1)

                    c_weight = torch.nan_to_num(
                        LA.norm(orig_norms.unsqueeze(-1) * alt_gradients + alt_norms.unsqueeze(-1) * orig_gradients, dim=-1) / (2 * orig_norms * alt_norms),
                        nan=0, posinf=0, neginf=0
                    )

                    # seq_len
                    plausibility_scores = torch.nan_to_num(
                        (1 - c_weight) * orig_norms / (orig_norms + alt_norms),
                        nan=1
                    )

                    kcomp = kldiv_loss(orig_probs, alt_probs)
                    kporig = kldiv_loss(plausible_probs, orig_probs)
                    kaorig = kldiv_loss(plausible_probs, alt_probs)


                    for indx in torch.nonzero(plausibility_scores > 0.1):
                        all_kldiv_indx.append(indx.item())
                        all_kldiv_losses.append(kporig[indx].item())
                        all_kldiv_alt_losses.append(kaorig[indx].item())
                        all_kldiv_comp_losses.append(kcomp[indx].item())

                    # if (plausibility_scores > 0.5).sum() >= 1:
                    #     return [orig_probs, alt_probs, plausible_probs, orig_gradients, alt_gradients, orig_norms, alt_norms]

                    resample_loss = torch.sum(orig_probs * (orig_probs.log() - plausible_probs.log()), dim=-1)
                    # alt_resample_loss = torch.sum(alt_probs * (alt_probs.log() - plausible_probs.log()), dim=-1)
                    alt_resample_loss = torch.square(LA.norm(alt_probs - plausible_probs, dim=-1))
                
                    # kl_loss(q, p) is KL_div (p || q) = E_p[log q]
                    # here i just use the loss of the alternate estimators wrt. the exact estimator

                    # get plausibility score for this sequence for particular head
                    # seq_len
                    head_plausibility_scores.append(plausibility_scores)
                    head_c_weight.append(c_weight)
                    head_resample_loss.append(resample_loss)
                    head_alt_resample_loss.append(alt_resample_loss)
                
                # concatenate plausibility scores for all heads, and add it to batch
                # seq_id n_heads seq_len
                if batch_plausibility_scores is None:
                    batch_plausibility_scores = torch.stack(head_plausibility_scores, dim=0).unsqueeze(0)
                    batch_c_weight = torch.stack(head_c_weight, dim=0).unsqueeze(0)
                    batch_resample_loss = torch.stack(head_resample_loss, dim=0).unsqueeze(0)
                    batch_alt_resample_loss = torch.stack(head_alt_resample_loss, dim=0).unsqueeze(0)
                else:
                    batch_plausibility_scores = torch.cat([batch_plausibility_scores, torch.stack(head_plausibility_scores, dim=0).unsqueeze(0)], dim=0)
                    batch_c_weight = torch.cat([batch_c_weight, torch.stack(head_c_weight, dim=0).unsqueeze(0)], dim=0)
                    batch_resample_loss = torch.cat([batch_resample_loss, torch.stack(head_resample_loss, dim=0).unsqueeze(0)], dim=0)
                    batch_alt_resample_loss = torch.cat([batch_alt_resample_loss, torch.stack(head_alt_resample_loss, dim=0).unsqueeze(0)], dim=0)
                batch_loss.append(kl_div) 
            
            # update all plausibility scores by adding scores in this batch
            # seq_id n_heads seq_len
            if all_plausibility_scores is None:
                all_plausibility_scores = batch_plausibility_scores
                all_c_weights = batch_c_weight
                all_resample_loss = batch_resample_loss
                all_alt_resample_loss = batch_alt_resample_loss
                all_losses = torch.stack(batch_loss, dim=0)
            else:
                all_plausibility_scores = torch.cat([all_plausibility_scores, batch_plausibility_scores], dim=0)
                all_c_weights = torch.cat([all_c_weights, batch_c_weight], dim=0)
                all_losses = torch.cat([all_losses, torch.stack(batch_loss, dim=0)], dim=0)
                all_resample_loss = torch.cat([all_resample_loss, batch_resample_loss], dim=0)
                all_alt_resample_loss = torch.cat([all_alt_resample_loss, batch_alt_resample_loss], dim=0)
            # kl_div = kl_loss(alt_probs, orig_probs.log(), reduction="none")
        
        if epoch % 20 == 19:
            path = "outputs/gauss/forward_s"
            with open(f"{path}/p_scores_{files_saved}.pkl", "wb") as f:
                pickle.dump(all_plausibility_scores.cpu(), f)
            with open(f"{path}/c_weights_{files_saved}.pkl", "wb") as f:
                pickle.dump(all_c_weights.cpu(), f)
            with open(f"{path}/loss_{files_saved}.pkl", "wb") as f:
                pickle.dump(all_losses.cpu(), f)
            with open(f"{path}/resample_loss_{files_saved}.pkl", "wb") as f:
                pickle.dump(all_resample_loss.cpu(), f)
            with open(f"{path}/alt_resample_loss_{files_saved}.pkl", "wb") as f:
                pickle.dump(all_alt_resample_loss.cpu(), f)
            
            big_arr = np.array([all_kldiv_indx, all_kldiv_losses, all_kldiv_alt_losses, all_kldiv_comp_losses])

            # ignore zeros
            sub_big_arr = big_arr[:,np.nonzero([big_arr[0,x] > 0 for x in range(big_arr.shape[1])])[0]]
            with open(f"{path}/subset_{files_saved}.pkl", "wb") as f:
                pickle.dump(sub_big_arr, f)

            files_saved += 1
            all_plausibility_scores = None
            all_c_weights = None
            all_resample_loss = None
            all_losses = None
            all_kldiv_indx = []
            all_kldiv_losses = []
            all_kldiv_alt_losses = []
            all_kldiv_comp_losses = []


# %%

def kldiv_loss(guess, truth):
    return torch.sum(truth * (truth.log() - guess.log()), dim=-1)

# %%

# big_arr = []
# for x in [all_kldiv_indx, all_kldiv_losses, all_kldiv_alt_losses, all_kldiv_comp_losses]:
#     little_arr = []
#     for y in x:
#         little_arr.append(y.item())
#     big_arr.append(little_arr)
# big_arr = np.array(big_arr)

# # ignore zeros
# sub_big_arr = big_arr[:,np.nonzero([big_arr[0,x] > 0 for x in range(big_arr.shape[1])])[0]]
# %%

big_arr[big_arr]
# %%

[orig_probs, alt_probs, plausible_probs, orig_gradients, alt_gradients, orig_norms, alt_norms] = run_plausibility_scoring()

# %%

# average over seq_id and seq_len to get by-head importance
total_plausibility = torch.sum(all_plausibility_scores[:,:,18], dim=[0])
importance_metric = torch.sum(all_plausibility_scores[:,:,18] * all_losses.unsqueeze(1)[:,:,18], dim=[0]) / total_plausibility
#     # sum of loss over the entire sequence



# # logits shape: n_heads seq_len vocab_size
# def sanity_check(model):
#     for batch in toxic_data_loader:
#         logits = model(batch, batch)
# # 
# %%

def export_csv(tensor, name):
    pd.DataFrame(tensor.reshape((12,12)).cpu().numpy()).stack().reset_index().to_csv(name)
def export_text(tensor, name):
    np.savetxt(name, tensor.cpu().numpy())
# %%

path = "outputs/reverse_s"
with open("all_losses.pkl", "wb") as f:
    pickle.dump(all_losses, f)
with open("all_losses.pkl", "wb") as f:
    pickle.dump(all_losses, f)
# %%
