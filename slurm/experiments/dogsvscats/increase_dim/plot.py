import os 
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

name = "catsvsdogs_increase_dim"
form = "%s*" % name

filenames = glob.glob(os.path.join("logs", form, "site_0", "logs", "train.csv"))
rows = []
full_rows = []
for filename in filenames:
    filename_only = filename.replace("logs/", "").replace("/site_0/", "").replace("train.csv", "")
    if "dm-dsgd" in filename_only:
        method = "dSGD"
    elif "dm-rankdad" in filename_only:
        method = "rank-dAD"
    elif "dm-dad" in filename_only:
        method = "dAD"
    elif "dm-topk" in filename_only:
        method = "top-3"
    if "m-vit1024" in filename_only:
        dim = 1024
    elif "m-vit512" in filename_only:
        dim = 512
    elif "m-vit256" in filename_only:
        dim = 256
    elif "m-vit128" in filename_only:
        dim = 128
    if "s-2" in filename_only:
        sites = 2
    elif "s-4" in filename_only:
        sites = 4
    elif "s-8" in filename_only:
        sites = 8
    elif "s-10" in filename_only:
        sites = 10
    elif "s-12" in filename_only:
        sites = 12
    elif "s-14" in filename_only:
        sites = 14
    elif "s-16" in filename_only:
        sites = 16
    elif "s-18" in filename_only:
        sites = 18
    df = pd.read_csv(filename)
    if np.max(df['step']) < 10:
        continue
    r = sum(df["_timer/batch_time"])
    cr = np.cumsum(df["_timer/batch_time"])
    row = dict(sites=sites, cumulative_runtime=r, method=method, dim=dim)
    for epoch, rr in enumerate(cr):
        full_row = copy.deepcopy(row)
        full_row["cumulative_runtime"] = rr
        full_row["epoch"] = epoch
        full_rows.append(full_row)
    rows.append(row)

df = pd.DataFrame(rows)
full_df = pd.DataFrame(full_rows)
sb.set()
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
DIMS = full_df["dim"].unique()
sb.lineplot(x="dim", y="cumulative_runtime", hue="method", data=df, ax=ax[0])
plt.xticks(DIMS)
sb.lineplot(x="dim", y="cumulative_runtime", hue="method", data=df, ax=ax[1])
plt.xticks(DIMS)
ax[1].set_yscale("log")
#plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.savefig(os.path.join("plots", name + ".png"), bbox_inches="tight")

DIMS = full_df["dim"].unique()
for DIM in DIMS:
    match_df = full_df[full_df["dim"] == DIM]

    sb.set()
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sb.lineplot(x="epoch", y="cumulative_runtime", hue="method", data=match_df, ax=ax[0])
    plt.ylabel("Cumulative Runtime")
    sb.lineplot(x="epoch", y="cumulative_runtime", hue="method", data=match_df, ax=ax[1])
    plt.ylabel("Cumulative Runtime")
    ax[1].set_yscale("log")
    plt.savefig(os.path.join("plots", name + "_dim=%s.png" % (DIM)), bbox_inches="tight")
