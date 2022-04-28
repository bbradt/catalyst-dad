import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

name = "catsvsdogs_increase_sites"
form = "%s*" % name
valid = "valid"
filenames = glob.glob(os.path.join("logs", form, "site_0", "logs", "%s.csv" % valid))
rows = []
full_rows = []
for filename in filenames:
    filename_only = filename.replace("logs/", "").replace("/site_0/", "").replace("%s.csv" % valid, "")
    valid_filename = filename.replace("train.csv", "valid.csv")
    if "dm-dsgd" in filename_only:
        method = "dSGD"
    elif "dm-rankdad" in filename_only:
        method = "rank-dAD"
    elif "dm-dad" in filename_only:
        method = "dAD"
        continue
    elif "dm-topk" in filename_only:
        method = "top-3"
    else:
        continue
    if "_s-2" in filename_only:
        sites = 2
    elif "_s-4" in filename_only:
        sites = 4
    elif "_s-6" in filename_only:
        sites = 6
    elif "_s-8" in filename_only:
        sites = 8
    elif "_s-10" in filename_only:
        sites = 10
    elif "_s-12" in filename_only:
        continue
        sites = 12
    elif "_s-14" in filename_only:
        sites = 14
    elif "_s-16" in filename_only:
        sites = 16
    elif "_s-18" in filename_only:
        sites = 18
    if 'k-0' in filename_only:
        k = 0
    elif 'k-1' in filename_only:
        k = 1
    elif 'k-2' in filename_only:
        k = 2
    elif 'k-3' in filename_only:
        k = 3
    elif 'k-4' in filename_only:
        k = 4
    else:
        continue
    df = pd.read_csv(filename)
    df = df.iloc[:30]
    r = sum([float(s) for s in df["cumulative_runtime"] if s != "cumulative_runtime" ])
    a = np.max(df["auc"])
    ac = np.max(df["accuracy"])
    row = dict(sites=sites, cumulative_runtime=r, method=method, auc=a, k=k, acc=ac)
    rows.append(row)
    rr = None
    if method == 'dSGD' and sites==4:
        print(filename)
        pass
    for i in range(len(df["cumulative_runtime"])):
        if df['auc'][i] == 'auc':
            continue
        r = df["cumulative_runtime"][i]
        if rr is None:
            rr = r
        else:
            rr += r
        a = float(df["auc"][i])
        ac = float(df['accuracy'][i])
        row = dict(sites=sites, cumulative_runtime=rr, method=method, auc=a, epoch=i, k=k, acc=ac)
        if method == 'dSGD' and sites==4:
            print(i)
        full_rows.append(row)
    

df = pd.DataFrame(rows)
full_df = pd.DataFrame(full_rows)

match_rows = []

for sites in [4, 8, 10, 14, 16]:
    site_df = full_df[full_df['sites'] == sites]
    for k in [0, 1, 2, 3, 4]:
        k_df = site_df[site_df['k'] == k]
        rankdad_df = k_df[k_df['method'] == 'rank-dAD']
        dsgd_df = k_df[k_df['method'] == 'dSGD']
        topk_df = k_df[k_df['method'] == 'top-3']
        baseline_auc = np.max(topk_df['auc'])
        if np.sum(rankdad_df['auc'] > baseline_auc).tolist() == 0:
            rankdad_index = -1
        else:
            rankdad_index=np.argwhere((rankdad_df['auc'] > baseline_auc).tolist())[0][0]
        if np.sum(dsgd_df['auc'] > baseline_auc).tolist() == 0:
            dsgd_index = -1
        else:
            dsgd_index=np.argwhere((dsgd_df['auc'] > baseline_auc).tolist())[0][0]
        if np.sum(topk_df['auc'] > baseline_auc).tolist() == 0:
            topk_index = -1
        else:
            topk_index=np.argwhere((topk_df['auc'] > baseline_auc).tolist())[0][0]
        rankdad_runtime=rankdad_df.iloc[rankdad_index]['cumulative_runtime']
        topk_runtime=topk_df.iloc[topk_index]['cumulative_runtime']
        dsgd_runtime=dsgd_df.iloc[dsgd_index]['cumulative_runtime']
        print("Sites, K ", sites, k)
        print("rank,top,dsgd ", rankdad_runtime, topk_runtime, dsgd_runtime)
        rank_row = dict(sites=sites, k=k, method="rank-dAD", time=rankdad_runtime)
        dsgd_row = dict(sites=sites, k=k, method="dSGD", time=dsgd_runtime)
        topk_row = dict(sites=sites, k=k, method="top-K",time=topk_runtime)
        match_rows.append(rank_row)
        match_rows.append(dsgd_row)
        match_rows.append(topk_row)

match_df = pd.DataFrame(match_rows)
sb.set(font_scale=2)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x='sites', y="time", hue="method", data=match_df, ax=ax, marker='o', markersize=12)
plt.ylabel("Time Taken to Match Best Top-3 AUC (Seconds)")
plt.xlabel("Number of Sites")
plt.savefig(os.path.join("plots", name + "_match_runtime.pdf"), bbox_inches="tight")


sb.set(font_scale=2)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="sites", y="cumulative_runtime", hue="method", data=df, ax=ax, marker="o", markersize=12)
plt.ylabel("Cumulative Runtime")
plt.xlabel("Sites")
plt.savefig(os.path.join("plots", name + "_runtime_linear.pdf"), bbox_inches="tight")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="sites", y="cumulative_runtime", hue="method", data=df, ax=ax, marker="o", markersize=12)
plt.ylabel("Cumulative Runtime")
plt.xlabel("Sites")
ax.set_yscale("log")
plt.savefig(os.path.join("plots", name + "_runtime_log.pdf"), bbox_inches="tight")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="sites", y="auc", hue="method", data=full_df, ax=ax, marker="o", markersize=12)
plt.ylabel("AUC")
plt.xlabel("Sites")
plt.savefig(os.path.join("plots", name + "_auc.pdf"), bbox_inches="tight")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sb.lineplot(x="sites", y="acc", hue="method", data=full_df, ax=ax, marker="o", markersize=12)
plt.ylabel("Accuracy")
plt.xlabel("Sites")
plt.savefig(os.path.join("plots", name + "_accuracy.pdf"), bbox_inches="tight")


sub_df = full_df[full_df['sites'] == 16]
max_dsgd = np.max(sub_df[sub_df['method'] == 'dSGD']['auc'])
maxes = None
for sites in [2, 4, 6, 8, 10, 14, 16]:
    sub_df = full_df[full_df['sites'] == sites]
    for k in range(5):
        sub_k_df = sub_df[sub_df['k'] == k]
        max_dsgd = sub_k_df[sub_k_df['method'] == 'dSGD']['auc']
        if maxes is None:
            maxes = max_dsgd.values
        else:
            maxes += max_dsgd.values
max_dsgd = np.max(maxes/5)
topkdad_df = sub_df[sub_df['method'] == 'top-3']
dsgd_df = sub_df[sub_df['method'] == 'dSGD']
rankdad_df = sub_df[sub_df['method'] == 'rank-dAD']

#rankdad_df = rankdad_df[rankdad_df['auc'] < max_dsgd]
rankdad_df = rankdad_df[rankdad_df['epoch'] < 9]
topkdad_df = sub_df[sub_df['method'] == 'top-3']
all_dfs = pd.concat([rankdad_df, topkdad_df, dsgd_df]).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sb.lineplot(x="epoch", y="auc", hue="method", data=all_dfs, ax=ax, marker="o", markersize=12)
plt.plot(range(30), max_dsgd*np.ones((30,)), 'k--')
plt.ylabel("AUC")
plt.xlabel("Epochs")
plt.savefig(os.path.join("plots", name + "_aligned_auc.pdf"), bbox_inches="tight")

all_sites = list(full_df['sites'].unique())
fig2, axes = plt.subplots(1, len(all_sites), figsize=(5* len(all_sites), 10), sharey=True)
for i, sites in enumerate(all_sites):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sub_df = full_df[full_df["sites"] == sites]
    sb.lineplot(x="epoch", y="auc", hue="method", data=sub_df, ax=ax, marker="o", markersize=12)
    plt.ylabel("AUC")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join("plots", name + "_oneexp_auc_s=%d.pdf" % sites), bbox_inches="tight")
    sb.lineplot(x="epoch", y="auc", hue="method", data=sub_df, ax=axes[i], marker="o", markersize=12)
    axes[i].set_title("Sites = %d" % sites)
    axes[i].set_ylim([0.5, np.max(full_df['auc']) + 0.1])
    if i != (len(all_sites) - 1):
        axes[i].get_legend().remove()
    plt.ylabel("AUC")
    plt.xlabel("Epoch")
fig2.savefig(os.path.join("plots", name + "_oneexp_auc.pdf"), bbox_inches="tight")