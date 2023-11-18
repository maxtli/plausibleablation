import pickle
import os
import torch

def save_files(obj,path="",i=None, cat=False):
    for k, v in obj.items():
        if i is None:
            fp = f"{path}{k}.pkl"
        else:
            fp = f"{path}{k}_{i}.pkl"
        if cat:
            v = torch.cat(v, dim=0).cpu()
        with open(fp, "wb") as f:
            pickle.dump(v,f)
            
def load_files(lst,path=""):
    ret = []
    for v in lst:
        fp = f"{path}{v}.pkl"
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                ret.append(pickle.load(f))
        else:
            i = 0
            fp = f"{path}{v}_{i}.pkl"
            ret_item = []
            while os.path.exists(fp):
                with open(fp, "rb") as f:
                    ret_item.append(pickle.load(f))
                i += 1
                fp = f"{path}{v}_{i}.pkl"
            if i == 0:
                print(fp, "file does not exist")
            ret.append(ret_item)
    return ret
