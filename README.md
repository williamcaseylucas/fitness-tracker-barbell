## Data Science Project Template

- Based on Dave Ebbelaar Research Project

Template for structuring your Python data science projects. It is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

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
  -
