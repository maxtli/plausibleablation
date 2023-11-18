# %%

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# %%
path = "outputs/long/forward_s"
with open(f"{path}/all_c.pkl", "rb") as f:
    all_c = pickle.load(f)
with open(f"{path}/all_losses.pkl", "rb") as f:
    all_losses = pickle.load(f)
with open(f"{path}/alll_plausibility_scores.pkl", "rb") as f:
    all_plausibility_scores = pickle.load(f)

# with open(f"{path}/all_c.pkl", "rb") as f:
#     all_c = pickle.load(f)
# with open(f"{path}/all_losses.pkl", "rb") as f:
#     all_losses = pickle.load(f)
# with open(f"{path}/alll_plausibility_scores.pkl", "rb") as f:
#     all_plausibility_scores = pickle.load(f)
# %%

sns.histplot(all_plausibility_scores.reshape(-1, all_plausibility_scores.shape[-1]))

# %%
all_plausibility_scores.shape
# %%

sns.histplot(all_c.reshape(-1, all_c.shape[-1]))


# %%
aps = torch.tensor(all_plausibility_scores)
acs = torch.tensor(all_c)
als = torch.tensor(all_losses)
total_plausibility = torch.sum(aps[:,:,18], dim=[0])
importance_metric = torch.sum(aps[:,:,18] * als.unsqueeze(1)[:,:,18], dim=[0]) / total_plausibility

# %%
pd.DataFrame(total_plausibility.reshape((12,12)))

# %%
total_plausibility
# %%
sns.histplot(all_plausibility_scores.flatten(), bins=100)
# %%
path = "outputs"
with open(f"{path}/zablation_losses.pkl", "rb") as f:
    zablation_losses = pickle.load(f)
with open(f"{path}/resample_losses.pkl", "rb") as f:
    resample_losses = pickle.load(f)
# %%
pd.DataFrame(zablation_losses[:,18].reshape((12,12)))
# %%
pd.DataFrame(resample_losses[:,18].reshape((12,12)))


# %%
