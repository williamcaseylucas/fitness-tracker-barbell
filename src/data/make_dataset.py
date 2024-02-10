import pandas as pd
from glob import glob
import numpy as np

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)  # 187

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion/"
f = files[0]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1

for f in files:

    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyro_df["epoch (ms)"]
del gyro_df["time (01:00)"]
del gyro_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
data_path = "../../data/raw/MetaMotion/"

def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:

        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]

    return acc_df, gyro_df


acc_df, gyro_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# set1 = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
# set2 = pd.concat([acc_df, gyro_df.iloc[:, :3]], axis=1)

# c1 = set1.count()
# c2 = set2.count()

# c1 + c2

# set1.combine
# merged_label = set1.combine_first(set2)
# merged_label.count()

# We don't need participant, label, category, and set to be duplicated

data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
# data_merged["set"] = merged_label

# data_merged["set"].count()  # Good!

# -- Doesn't work the same, get fewer rows and lose datetime --
# pd.merge(
#     acc_df, gyro_df, how="outer"
# )

# -- removes from 69,000 to 1,000. Yikes! --
# data_merged.dropna()

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# data_merged[:100].resample(rule="S").mean()
# data_merged.resample(rule="200ms").mean()  # 5 measurements per second

# for categorical, grab the last in that data range
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# delta = (
#     data_merged[data_merged["set"] == 1].index[-1]
#     - data_merged[data_merged["set"] == 1].index[0]
# )
# acc_delta = acc_df[acc_df["set"] == 1].index[-1] - acc_df[acc_df["set"] == 1].index[0]
# gyr_delta = (
#     gyro_df[gyro_df["set"] == 1].index[-1] - gyro_df[gyro_df["set"] == 1].index[0]
# )


# data_merged[data_merged["set"] == 1].resample(rule="200ms").apply(
#     sampling
# ).isnull().sum()

# Could have just done this
# data_resampled = data_merged.resample(rule="200ms").apply(sampling).dropna()

# Split by day (n -> day timeframes, g -> df group)
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

# data_merged[:100].resample(rule="S").mean()
# data_merged.resample(rule="200ms").mean()  # 5 measurements per second

data_resampled[data_resampled["set"] == 1]
sorted(data_resampled["set"].unique())

data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
