# %%
import seaborn as sns
from plausibility_utils import load_files
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%

alt_orig_ests, orig_alt_ests, patched_alt_ests = load_files(["alt_orig_ests", "orig_alt_ests","patched_alt_ests"],path="ioi/simple/")

# %%

alt_orig_ests = torch.cat(alt_orig_ests, dim=0)
orig_alt_ests = torch.cat(orig_alt_ests, dim=0)
patched_alt_ests = torch.cat(patched_alt_ests, dim=0)

# %%

def gauss_kernel(ts,sigma):
    return torch.exp(-1 * torch.square(ts) / (2 * sigma * sigma))

def get_importance(loss, alt_loss, sigma_low=1e-2, sigma_high=1, step=1e-2):
    sigmas = []
    imp_tensors = []
    apx_samp_size = []
    tpl = []
    for i in tqdm(range(int((sigma_high - sigma_low) // step))):
        sigma = sigma_low + i * step
        p_weights = gauss_kernel(loss,sigma)
        imp_tensors.append(torch.sum(p_weights * alt_loss, dim=0) / torch.sum(p_weights, dim=0))
        apx_samp_size.append(1 / torch.sqrt(torch.sum(torch.square(p_weights / torch.sum(p_weights, dim=0)),dim=0)))
        sigmas.append(sigma)
        tpl.append(torch.sum(p_weights,dim=0))
    
    max_impt = [torch.max(x).item() for x in imp_tensors]
    mean_impt = [torch.median(x).item() for x in imp_tensors]
    std_impt = [torch.quantile(x,.75).item() for x in imp_tensors]
    std_low_impt = [torch.quantile(x,.25).item() for x in imp_tensors]
    return sigmas, imp_tensors, max_impt, mean_impt, std_impt, std_low_impt, tpl, apx_samp_size

# %%

sigmas, imp_tensors, max_impt, mean_impt, stdp, stdm, tpl, apx_samp_size = get_importance(patched_alt_ests, alt_orig_ests.unsqueeze(1))


# %%

ax1=sns.lineplot(x=sigmas, y=max_impt, label="max")
sns.lineplot(x=sigmas, y=mean_impt, label="median")
sns.lineplot(x=sigmas, y=stdp, label="75th")
sns.lineplot(x=sigmas, y=stdm, label="25th")

shp = patched_alt_ests[0].shape
max_samp = [torch.max(x).item() / shp[0] for x in tpl]
mean_samp = [torch.median(x).item() / shp[0] for x in tpl]
std_samp = [torch.quantile(x,.75).item() / shp[0] for x in tpl]
std_low_samp = [torch.quantile(x,.25).item() / shp[0] for x in tpl]

ax1=sns.lineplot(x=sigmas, y=max_samp, label="max", ax=ax1.twinx())
sns.lineplot(x=sigmas, y=mean_samp, label="median")
sns.lineplot(x=sigmas, y=std_samp, label="75th")
sns.lineplot(x=sigmas, y=std_low_samp, label="25th")

ax1.legend()

# %%

rel_idx = 20
rel_impt = imp_tensors[rel_idx].reshape((12,12))
rel_tp = tpl[20]

layer_id = torch.arange(rel_impt.shape[0]).unsqueeze(1).repeat(1,rel_impt.shape[1])
head_id = torch.arange(rel_impt.shape[1]).repeat(rel_impt.shape[0],1)

print(layer_id)
print(head_id)

# %%

rel_idxes = [10,20,50]
comp_impt = pd.DataFrame({"layer_id": layer_id.flatten().numpy(), "head_id": head_id.flatten().numpy(), **{f"imptc_{i}": imp_tensors[i] for i in rel_idxes}, **{f"log_samp_{i}": torch.log(tpl[i]) for i in rel_idxes}, **{f"samp_sz_{i}": apx_samp_size[i] for i in rel_idxes}})

# %%

display(comp_impt.sort_values("imptc_20").tail(20))

# %%
# print(sns.histplot(rel_impt))
# print(torch.log(rel_tp))

# important_heads = torch.nonzero((rel_impt > 2))
# print(rel_impt[important_heads[:,0],important_heads[:,1]])
# print()


# %%

# for dim in [3,5,7,10,15,20,30,35]:
#     plt.figure()
#     plt.title(f"Neuron importances with {dim} output dimensions")
#     (sigmas, ts, max, mean, stdp, stdm, tpl), shp = get_importances_for_dims(dim)
#     ax1=sns.lineplot(x=sigmas, y=max, label="max")
#     sns.lineplot(x=sigmas, y=mean, label="median")
#     sns.lineplot(x=sigmas, y=stdp, label="75th")
#     sns.lineplot(x=sigmas, y=stdm, label="25th")

#     max_samp = [torch.max(x).item() / shp[0] for x in tpl]
#     mean_samp = [torch.median(x).item() / shp[0] for x in tpl]
#     std_samp = [torch.quantile(x,.75).item() / shp[0] for x in tpl]
#     std_low_samp = [torch.quantile(x,.25).item() / shp[0] for x in tpl]

#     ax1=sns.lineplot(x=sigmas, y=max_samp, label="max", ax=ax1.twinx())
#     sns.lineplot(x=sigmas, y=mean_samp, label="median")
#     sns.lineplot(x=sigmas, y=std_samp, label="75th")
#     sns.lineplot(x=sigmas, y=std_low_samp, label="25th")

#     ax1.legend()
#     plt.savefig(f"results/fig_{dim}.png")

#     layers = [96,192,384,768]
#     start_idx = 0
#     for i,layer_dim in enumerate(layers):
#         plt.figure()
#         plt.title(f"Neuron importances after layer {i} with {dim} output dimensions")
#         layer_ts = [x[start_idx:start_idx+layer_dim] for x in ts]
#         layer_tpl = [x[start_idx:start_idx+layer_dim] for x in tpl]
#         max_impt = [torch.max(x).item() for x in layer_ts]
#         mean_impt = [torch.median(x).item() for x in layer_ts]
#         std_impt = [torch.quantile(x,.75).item() for x in layer_ts]
#         std_low_impt = [torch.quantile(x,.25).item() for x in layer_ts]
#         ax1=sns.lineplot(x=sigmas, y=max_impt, label="max")
#         sns.lineplot(x=sigmas, y=mean_impt, label="median")
#         sns.lineplot(x=sigmas, y=std_impt, label="75th")
#         sns.lineplot(x=sigmas, y=std_low_impt, label="25th")

#         max_samp = [torch.max(x).item() / shp[0] for x in layer_tpl]
#         mean_samp = [torch.median(x).item() / shp[0] for x in layer_tpl]
#         std_samp = [torch.quantile(x,.75).item() / shp[0] for x in layer_tpl]
#         std_low_samp = [torch.quantile(x,.25).item() / shp[0] for x in layer_tpl]

#         ax1=sns.lineplot(x=sigmas, y=max_samp, label="max", ax=ax1.twinx())
#         sns.lineplot(x=sigmas, y=mean_samp, label="median")
#         sns.lineplot(x=sigmas, y=std_samp, label="75th")
#         sns.lineplot(x=sigmas, y=std_low_samp, label="25th")
#         ax1.legend()

#         plt.savefig(f"results/fig_{dim}_layer_{i}.png")
#         start_idx += layer_dim


# # %%
# for dim in [3,5,7,10,15,20]:
#     (sigmas, ts, max, mean, stdp, stdm, tpl), shp = get_importances_for_dims(dim)

#     layers = [96,192,384,768]
#     start_idx = 0
#     for i,layer_dim in enumerate(layers):
#         plt.figure()
#         plt.title(f"Neuron importances self-correlation after layer {i} with {dim} output dimensions")
#         layer_ts = [x[start_idx:start_idx+layer_dim] for x in ts]
#         print(layer_ts[0].shape)
#         sns.scatterplot(x=(layer_ts[5]).numpy(), y=(layer_ts[10]).numpy(), s=5)
#         plt.savefig(f"results/self-corr/fig_{dim}_layer_{i}.png")


# %%
total_p = patched_alt_ests[0].sum(dim=0)
importances = (patched_alt_ests[0] * alt_orig_ests[0].unsqueeze(1)).sum(dim=0) / total_p
# %%
sns.histplot(importances.numpy().flatten())

# %%

torch.mean(alt_orig_ests[0],dim=0)
# %%
