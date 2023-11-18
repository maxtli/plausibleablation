# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# %%
path = "outputs/gauss/forward_s"
 
 # %%
i = 0
with open(f"{path}/c_weights_{i}.pkl", "rb") as f:
    all_c = pickle.load(f)
with open(f"{path}/loss_{i}.pkl", "rb") as f:
    all_losses = pickle.load(f)
with open(f"{path}/p_scores_{i}.pkl", "rb") as f:
    all_plausibility_scores = pickle.load(f)
with open(f"{path}/resample_loss_{i}.pkl", "rb") as f:
    resample_loss = pickle.load(f)
with open(f"{path}/alt_resample_loss_{i}.pkl", "rb") as f:
    alt_resample_loss = pickle.load(f)
with open(f"{path}/subset_{i}.pkl", "rb") as f:
    subset_analysis = pickle.load(f)
# %%

def plot_ts(ts):
    sns.histplot(ts[...,-1].flatten().cpu().numpy(), bins=100)

# %%
plot_ts(all_c)
# %%
plot_ts(all_losses)
# %%

comp_path = "outputs/long/forward_s"

i = 0
with open(f"{comp_path}/loss_2.pkl", "rb") as f:
    comp_l = pickle.load(f)
with open(f"{comp_path}/p_scores_2.pkl", "rb") as f:
    p_l = pickle.load(f)
with open(f"{comp_path}/c_weights_2.pkl", "rb") as f:
    comp_c = pickle.load(f)

comp_tp = torch.sum(p_l[:,:,18], dim=[0])
comp_movi = torch.sum(p_l[:,:,18] * comp_l.unsqueeze(1)[:,:,18], dim=[0]) / comp_tp


# %%
c_series = all_c[...,-1].flatten().cpu().numpy()
ps_series = pd.Series(all_plausibility_scores[...,-1].flatten().cpu().numpy() / (1 - c_series))
sns.scatterplot(x=c_series,y=np.log(ps_series), s=1)
# plt.ylim(-5,0)
# %%
# %%

total_plausibility = torch.sum(all_plausibility_scores[:,:,18], dim=[0])
importance_metric = torch.sum(all_plausibility_scores[:,:,18] * all_losses.unsqueeze(1)[:,:,18], dim=[0]) / total_plausibility

total_c = torch.sum(1-all_c[:,:,18], dim=[0])
new_importance_metric = torch.sum((1-all_c[:,:,18]) * resample_loss[:,:,18], dim=[0]) / total_c

# sns.histplot(np.log(ps_series))

# %%

def calc_effective_stats(p_s):
    effective_sample_size = torch.square(torch.sum(p_s[:,:,18], dim=[0])) / torch.sum(torch.square(p_s[:,:,18]), dim=[0])
    sample_entropy = torch.mean(torch.log(p_s[:,:,18]), dim=[0]) - torch.log(torch.sum(p_s[:,:,18], dim=[0]))
    return effective_sample_size, sample_entropy
# %%
ess, entropy = calc_effective_stats(all_plausibility_scores)
comp_ess, comp_entropy = calc_effective_stats(p_l)
# %%


# try Gaussian kernel
sns.histplot(alt_resample_loss.cpu().flatten())

def get_gaussian_likelihood(sqdist, sigma=.1):
    return torch.exp(-sqdist / (2 * sigma * sigma))
# %%
gauss_likelihood = get_gaussian_likelihood(alt_resample_loss, 1)
total_likelihood = torch.sum(gauss_likelihood[:,:,18], dim=[0])
importance_metric = torch.sum(gauss_likelihood[:,:,18] * all_losses.unsqueeze(1)[:,:,18], dim=[0]) / total_likelihood