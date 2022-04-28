import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

name = "cifar10_learning_rate"
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
        continue
    elif "dm-topk" in filename_only:
        method = "top-3"
    if "s-2" in filename_only:
        sites = 2
    elif "s-4" in filename_only:
        sites = 4
    elif "s-8" in filename_only:
        sites = 8
    elif "s-10" in filename_only:
        sites = 10
    elif "s-12" in filename_only:
        continue
        sites = 12
    elif "s-14" in filename_only:
        sites = 14
    elif "s-16" in filename_only:
        sites = 16
    elif "s-18" in filename_only:
        sites = 18
    df = pd.read_csv(filename)
    r = sum([float(s) for s in df["cumulative_runtime"] if s != "cumulative_runtime" ])
    a = np.max(df["auc"])
    if 'lr_' not in filename:
        continue
    lr = float(filename[(filename.index('lr_')+3):filename.index("_mk")])
    row = dict(sites=sites, cumulative_runtime=r, method=method, auc=a, learning_rate=lr)
    rows.append(row)
    for i in range(len(df["cumulative_runtime"])):
        if df['auc'][i] == 'auc':
            continue
        r = df["cumulative_runtime"][i]
        a = float(df["auc"][i])
        row = dict(sites=sites, cumulative_runtime=r, method=method, auc=a, epoch=i, learning_rate=lr)
        full_rows.append(row)

df = pd.DataFrame(rows)
full_df = pd.DataFrame(full_rows)
sb.set(font_scale=2)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="learning_rate", y="cumulative_runtime", hue="method", data=df, ax=ax, marker="o", markersize=12)
plt.ylabel("Cumulative Runtime")
plt.xlabel("Sites")
plt.savefig(os.path.join("plots", name + "_runtime_linear.png"), bbox_inches="tight")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="learning_rate", y="cumulative_runtime", hue="method", data=df, ax=ax, marker="o", markersize=12)
plt.ylabel("Cumulative Runtime")
plt.xlabel("Sites")
ax.set_yscale("log")
plt.savefig(os.path.join("plots", name + "_runtime_log.png"), bbox_inches="tight")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.boxplot(x="learning_rate", y="auc", hue="method", data=full_df, ax=ax)
plt.ylabel("AUC")
plt.xlabel("Learning Rate")
plt.savefig(os.path.join("plots", name + "_auc.png"), bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sub_df = full_df[full_df["sites"] == 4]
sb.lineplot(x="epoch", y="auc", hue="method", data=sub_df, ax=ax, marker="o", markersize=12)
plt.ylabel("AUC")
plt.xlabel("Epoch")
plt.savefig(os.path.join("plots", name + "_oneexp_auc.png"), bbox_inches="tight")

