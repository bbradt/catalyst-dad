import os
import glob
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

FOLDER_SPLIT_CHAR = ";"
VARIABLE_SPLIT_CHAR = "="
EXPERIMENT_DIR = os.path.join(
    "/data",
    "users2",
    "bbaker",
    "projects",
    "dist_autodiff",
    "logs",
    "dAD-ReTesting-MLP"
)
TRAIN_CSV_SUFFIX = os.path.join("site_0", "logs", "train.csv")
VALID_CSV_SUFFIX = os.path.join("site_0", "logs", "valid.csv")

def compile_results(experiment_dir=EXPERIMENT_DIR, train_csv_suffix=TRAIN_CSV_SUFFIX, valid_csv_suffix=VALID_CSV_SUFFIX):
    all_folders = glob.glob(os.path.join(EXPERIMENT_DIR, "nn=*"))
    experiment_train_dfs = []
    experiment_valid_dfs = []
    for experiment_folder in all_folders:
        experiment_train_csv = os.path.join(experiment_folder, TRAIN_CSV_SUFFIX)
        experiment_valid_csv = os.path.join(experiment_folder, VALID_CSV_SUFFIX)
        if not os.path.exists(experiment_train_csv):
            print("%s does not exist" % experiment_train_csv)
            continue
        experiment_train_df = pd.read_csv(experiment_train_csv)
        experiment_valid_df = pd.read_csv(experiment_valid_csv)
        experiment_basename = os.path.basename(experiment_folder)
        experiment_vars = experiment_basename.split(FOLDER_SPLIT_CHAR)
        for experiment_var in experiment_vars:
            experiment_key, experiment_val = experiment_var.split("=")
            try:
                experiment_val = eval(experiment_val)
                if type(experiment_val) in [list, dict]:
                    experiment_val = str(experiment_val)
            except Exception as e:
                pass
            experiment_train_df[experiment_key] = experiment_val
            experiment_valid_df[experiment_key] = experiment_val
        experiment_train_dfs.append(experiment_train_df)
        experiment_valid_dfs.append(experiment_valid_df)

    experiment_train_compiled_df = pd.concat(experiment_train_dfs).reset_index(drop=True)
    experiment_valid_compiled_df = pd.concat(experiment_valid_dfs).reset_index(drop=True)
    return experiment_train_compiled_df, experiment_valid_compiled_df


                
train_df, valid_df = compile_results()

fig, ax = plt.subplots(3, 2, figsize=(10, 10))
sb.lineplot(x="step", y="auc", data=train_df, ax=ax[0,0], hue="dm", style="be")
sb.lineplot(x="step", y="auc", data=valid_df, ax=ax[0,1], hue="dm", style="be")
ax[0,0].set_title("Training")
ax[0,1].set_title("Validation")
sb.lineplot(x="step", y="runtime", data=train_df, ax=ax[1,0], hue="dm", style="be")
sb.lineplot(x="step", y="runtime", data=valid_df, ax=ax[1,1], hue="dm", style="be")
sb.lineplot(x="step", y="bits", data=train_df, ax=ax[2,0], hue="dm", style="be")
sb.lineplot(x="step", y="bits", data=valid_df, ax=ax[2,1], hue="dm", style="be")

plt.savefig("figures/Initial-Retesting-MLP.png", bbox_inches="tight")
print('ok')