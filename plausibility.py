# %%
from model import load_demo_gpt2, tokenizer, DEVICE
from data import retrieve_toxic_data, retrieve_owt_data
import torch
from torch import linalg as LA
from tqdm import tqdm
from itertools import cycle

# %%
import seaborn as sns
import pandas as pd
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

# marginal resample ablation
all_losses = []
for epoch in tqdm(range(20)):
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

        orig_probs = probs[:,[1]]
        alt_probs = probs[:,2:]
        
        # seq_len
        ce_loss = torch.sum(-1 * alt_probs * orig_probs.log(), dim=-1)
        all_losses.append(ce_loss)

# %%
import pickle 
als = torch.cat(all_losses, dim=0).mean(dim=0).cpu().numpy()
with open("outputs/resample_losses.pkl", "wb") as f:
    pickle.dump(als, f)


# %%
# # zero ablation
# for c, batch in enumerate(toxic_data_loader):
#     # print("Batches complete:", batches)
#     # batches += 1

#     if batch.shape[0] < toxic_batch_size:
#         continue

#     # remove start token
#     # randomize the OWT batch
#     batch = batch[:,1:].long().to(DEVICE)

#     # shape: batch n_heads (0 alt, 1 original) seq_len vocab_size
#     with torch.no_grad():
#         logits = model(batch, torch.zeros(batch.shape).)
    
#     probs = torch.softmax(logits, dim=-1)

#     orig_probs = probs[:,[1]]
#     alt_probs = probs[:,2:]
    
#     # seq_len
#     ce_loss = torch.sum(-1 * alt_probs * orig_probs.log(), dim=-1)
#     all_losses.append(ce_loss)

# %%
all_plausibility_scores = None
all_c_weights = None
all_losses = None

for epoch in tqdm(range(20)):
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
        batch_loss = []
        
        # because batching requires too much memory
        for id_in_batch in range(probs.shape[0]):
            orig_probs = probs[id_in_batch,1]
            alt_probs = probs[id_in_batch,0]
            
            # seq_len
            ce_loss = torch.sum(-1 * alt_probs * orig_probs.log(), dim=-1)

            head_plausibility_scores = []
            head_c_weight = []

            for head in range(2, probs.shape[1]):
                plausible_probs = probs[id_in_batch,head]

                # orig_gradients = plausible_probs / orig_probs
                orig_gradients = orig_probs / plausible_probs
                orig_gradients -= torch.mean(orig_gradients, dim=-1).unsqueeze(-1)

                # seq_len
                orig_norms = LA.norm(orig_gradients, dim=-1)

                # alt_gradients = plausible_probs / alt_probs
                alt_gradients = alt_probs / plausible_probs
                alt_gradients -= torch.mean(alt_gradients, dim=-1).unsqueeze(-1)

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
            
                # kl_loss(q, p) is KL_div (p || q) = E_p[log q]
                # here i just use the loss of the alternate estimators wrt. the exact estimator

                # get plausibility score for this sequence for particular head
                # seq_len
                head_plausibility_scores.append(plausibility_scores)
                head_c_weight.append(c_weight)
            
            # concatenate plausibility scores for all heads, and add it to batch
            # seq_id n_heads seq_len
            if batch_plausibility_scores is None:
                batch_plausibility_scores = torch.stack(head_plausibility_scores, dim=0).unsqueeze(0)
                batch_c_weight = torch.stack(head_c_weight, dim=0).unsqueeze(0)
            else:
                batch_plausibility_scores = torch.cat([batch_plausibility_scores, torch.stack(head_plausibility_scores, dim=0).unsqueeze(0)], dim=0)
                batch_c_weight = torch.cat([batch_c_weight, torch.stack(head_c_weight, dim=0).unsqueeze(0)], dim=0)
            batch_loss.append(ce_loss) 
        
        # update all plausibility scores by adding scores in this batch
        # seq_id n_heads seq_len
        if all_plausibility_scores is None:
            all_plausibility_scores = batch_plausibility_scores
            all_c_weights = batch_c_weight
            all_losses = torch.stack(batch_loss, dim=0)
        else:
            all_plausibility_scores = torch.cat([all_plausibility_scores, batch_plausibility_scores], dim=0)
            all_c_weights = torch.cat([all_c_weights, batch_c_weight], dim=0)
            all_losses = torch.cat([all_losses, torch.stack(batch_loss, dim=0)], dim=0)
        # kl_div = kl_loss(alt_probs, orig_probs.log(), reduction="none")

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
