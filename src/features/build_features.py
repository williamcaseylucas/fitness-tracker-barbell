import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

# ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
predictor_columns = list(df.columns[:6])

# plot settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df.info()  # some outliers exist from removing them before

# Because our data set has a big time lapse, we have to reset the index
# subset = df[df["set"] == 35]["gyr_y"].reset_index(drop=True).plot()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()

duration = df[df["set"] == 2].index[-1] - df[df["set"] == 2].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start

    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

# Dividing by the number of repetitions
heavy_average_duration = duration_df.iloc[0] / 5  # heavy -> 2.94
medium_average_duration = duration_df.iloc[1] / 10  # medium -> 2.49

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# We want per second which is where 1000 ms comes from. 200 comes from our resampling rate
fs = 1000 / 200
cutoff = 1.3

# Creates 'acc_y_lowpass' column
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

# Visualize difference with low pass filter applied
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Override original columns
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Apply pca on first six columns
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(
    df_pca, predictor_columns, 3
)  # Captured 6 columns into three while capturing as much of the variance as possible

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

dfa_squared = df_pca.copy()

acc_r = (
    dfa_squared["acc_x"] ** 2 + dfa_squared["acc_y"] ** 2 + dfa_squared["acc_z"] ** 2
)
gyr_r = (
    dfa_squared["gyr_x"] ** 2 + dfa_squared["gyr_y"] ** 2 + dfa_squared["gyr_z"] ** 2
)

dfa_squared["acc_r"] = np.sqrt(acc_r)
dfa_squared["gyr_r"] = np.sqrt(gyr_r)

subset = dfa_squared[dfa_squared["set"] == 18]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = dfa_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = 1000 // 200  # gives us a window size of 1 second

# Problem with this: It is apply window over last 5 rows, but our rows have different sets incorporated into them. Thus, we need to split by the sets first, then apply the rolling average with window
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")


# Fixes the above issue
df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()

    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "mean")
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "std")

    # for col in predictor_columns:
    #     subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
    #     subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

# Recombine the sets again
df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = 1000 // 200
ws = 2800 // 200  # average length of a repition was about 2.8 seconds so we say 2800

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

df_freq.columns

subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()

subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq.isnull().sum()
df_freq.info()

df_freq.dropna(inplace=True)
any(val == 0 for val in list(df_freq.isnull().sum()))  # check no more null

# Skip every other row -> Important for windows to avoid overfitting
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []


for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)

    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# k = 5

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

df_cluster["cluster"].unique()  # see clusters

# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
