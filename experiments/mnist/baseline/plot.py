import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

name = "mnist_baseline"
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
    df = pd.read_csv(filename)
    r = 0
    for i, runtime in enumerate(df["cumulative_runtime"]):
        r += runtime
        row = dict(epoch=i, cumulative_runtime=r, runtime=runtime, method=method)
        rows.append(row)

df = pd.DataFrame(rows)
sb.set()
fig, ax = plt.subplots(1, 2)

sb.lineplot(x="epoch", y="cumulative_runtime", hue="method", data=df, ax=ax[0])
sb.lineplot(x="epoch", y="cumulative_runtime", hue="method", data=df, ax=ax[1])
ax[1].set_yscale("log")
plt.savefig(os.path.join("plots", name + ".png"), bbox_inches="tight")
