## Data Science Project Template

- Based on Dave Ebbelaar Research Project

Template for structuring your Python data science projects. It is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

## Notes

Better to export pickle if you are using df later so date time stuff is preserved

Filter subtle noise (not outliers) and identify parts of data that explain most of the variance

Add termporal, frequency, and cluster features

### Outlier Detection (IQR, Chauvenet, LOF)

- Better to grab exercises by label first, then apply outlier detection through boxplots, calculating IQR values, and plotting them via 'groupby'
- How to test for normal distribtuion -> boxplot has whiskers that are approx similar or histogram with bell shaped curve
- Chauvenet's Criterion -> Detects outliers, assumes normal distribtution
  - Rest data is the only one that does not look normally distributed
- IQR can sometimes be useful, sometimes Chauvenet's can be good if normal distribution is assumed
- Local Outlier Factor
  - Distance based
  - Unsupervized learning approach
  - Uses density of clusters with KNN to determine outliers
  - Can identify outliers within data itself using this approach

## Data

- Gyroscope data -> deg/s
  - more measurements per second
  - .08 seconds (1/12.5)
- Accelerometer data -> g force
  - .04 seconds (1/25)
- 5 participants did the following with a **light set** and a **heavy set** and a **rest set** with recordings being 5 repetitions
  - bench press
  - deadlift
  - overhead press
  - barbell row
  - squat
- Supervized learning project
  - Will consist of structured and unstructured data
  - Multi-class classification goal (6 in total)
- Files are mostly unstructured so they have to be stiched together as we incorprate them into Pandas df's
- Want
  - Timestamp
  - x, y, z accel
  - x, y, z gyro
  - label indicating exercise type (or rest)
- File naming convention (split on '-')
  - Participant
  - Label: type of activity ('overhead press would be ohp')
  - category (heavy, medium, standing, etc)
- Adding 'set' to each df so that we don't have to group by category, label, and participant later
- Epochs -> Unit Time
  - Time since Jan 1, 1970
- #### Visuals
  ##### Comparing Participants
  - ![Comparing Participants](./reports/figures/subset_comparing_participants.png)
  ##### x, y, z accel. data for participant
  - ![x, y, z accel. data for participant](./reports/figures/xyz_for_participant.png)
  ##### Medium vs Heavy Set
  - ![Medium vs Heavy Set](./reports/figures/medium_vs_heavy.png)

### Feature Engineering

- Can drop bad rows, or use imputation (mean, median, min, max between two values, or interpolate)
- Used Butterworth lowpass filter

  - Frequency is 1000/200 since 1000 ms is 1 second and 200ms was the rate we we resampled at
  - cutoff -> higher the number, more angles present, lower the number, smoother the curves
  - Used to remove high frequency noise from a dataset
  - Removes any data points above a certain threshold frequency, while still preserving the underlying pattern of the data

- PCA

  - Passing in first six columns into PCA method and then using the 'elbow' method to analyze
  - Elbow occurred at 3 labels
  - Captured 6 columns into three while capturing as much of the variance as possible

- Sum of squares (magnitude)

  - Direction is impartial to device orientation and can handle dynamic re-orientations

- Fourier Transformation

  - Measurements to be represented by sinusoid functions with different frequencies
  - Data can be represented as frequency components
  - Provides insights into patterns and trends that may otherwise not have been visible
  - DFT (Discrete Fourier Transformation) can help reduce noise allowing for more accurate models

- Temporal Abstraction

  - Using a rolling window walking over our data to compute, over a window size, the average and standard deviation
  - Will use mean and standard deviation

- Windows

  - For windows, we skipped every other column to avoid overfitting due to aggregated new values in our windows

- Clustering

  - KMeans can use elbow method as well
  - Cluster 0 combines bench press and overhead press

  ##### KMeans Clusters Visualization

  - ![KMeans Clusters Visualization](./reports/figures/kmeans_clusters.png)

  ##### KMeans Labels Visualization

  - ![KMeans Labels Visualization](./reports/figures/kmeans_labels.png)

  ## Training

  - Basic Features
    Square Features
    PCA Features
    Time Features
    Frequency Features
    Cluster Features

  ##### Feature Importance w/ Accuracy

  - ![Feature Importance w/ Accuracy](./reports/figures/accuracy_with_features.png)
  - Use grid search and K-fold cross validation
  - What Barchart tells us
    ##### Feature Set Accuracies
    - ![Feature Set Accuracies](./reports/figures/featuresets_accuracy_barchart.png)
    - Feature set 4 is almost the best for all of the models which includes all of our features
    - The selected features compare better in all of the cases compared to feature set 1 and feature set 2
  - Use Confusion matrix to highlight what best performing model looks like
    ##### Confusion Matrix
    - ![Confusion Matrix](./reports/figures/confusion_matrix.png)
    - Window aspect may be contributing to some of the error since some of this training looks similar

### Counting Repetitions

- Heavy -> 5 repetitions per set
- Medium -> 10 repetitions per set
- After applying filter to count repetitions, we can see the peaks of 'acc_y' are of interest
  ##### Without lowpass filter
  - ![Without lowpass filter](./reports/figures/without_lowpass_filter.png)
  ##### With lowpass filter
  - ![With lowpass filter](./reports/figures/with_lowpass_filter.png)
- Use argrelextrema to find relative extrema which will tell us our peaks
- MAE
  ##### Mean Average Error for Counts
  - ![MAE](./reports/figures/mae_of_miscategorizations.png)
  - To improve, isolate each exercise, create a specific model for each of them and tweak 'order' parameter along with others to get as exact as possible
