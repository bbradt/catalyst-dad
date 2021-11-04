import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

name = "mnist_increase_sites"
form = "%s*" % name

filenames = glob.glob(os.path.join("logs", form, "site_0", "logs", "train.csv"))
rows = []
for filename in filenames:
    filename_only = filename.replace("logs/", "").replace("/site_0/", "").replace("train.csv", "")
    if "dm-dsgd" in filename_only:
        method = "dSGD"
    elif "dm-rankdad" in filename_only:
        method = "rank-dAD"
    elif "dm-dad" in filename_only:
        method = "dAD"
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
    df = pd.read_csv(filename)
    r = sum(df["cumulative_runtime"])
    row = dict(sites=sites, cumulative_runtime=r, method=method)
    rows.append(row)

df = pd.DataFrame(rows)
sb.set()
fig, ax = plt.subplots(1, 2)

sb.lineplot(x="sites", y="cumulative_runtime", hue="method", data=df, ax=ax[0])
sb.lineplot(x="sites", y="cumulative_runtime", hue="method", data=df, ax=ax[1])
ax[1].set_yscale("log")
plt.savefig(os.path.join("plots", name + ".png"), bbox_inches="tight")
