# Project Workflow

The scripts must be executed in the following order:

1. **photo_collector.py** – Collect gesture images using the webcam.
2. **creating_dataset.py** – Extract hand landmarks from the collected images and generate the dataset.
3. **train_classifier.py** – Train the Random Forest classifier using the generated dataset.
4. **test_classifier.py** – Test the trained model with real-time webcam prediction.

## Data Collection Guidelines

This project uses **your own hand images** to build a custom dataset. A reference image is included in the repository to help you reproduce each sign consistently during data collection.

For the best results:

- Hold the hand gesture **steady** while collecting images.
- Do **not** rotate or change the hand pose during capture.
- Move your hand **only toward and away from the camera** (forward and backward) to capture different scales.
- Try to keep the gesture centered in the camera frame.
- Ensure the hand remains clearly visible and is not occluded.
- Collect images under slightly different distances and lighting conditions to improve the model's robustness.

Following these recommendations will produce a cleaner dataset and improve the classifier's overall accuracy.


# Photo Collector

This code collects images of hand gesture from webcam (the first connected) and automatically organizes them into a folder named photo_collection beside the project folder.

## Features
- Captures images directly from the webcam.
- Automatically creates dataset folders.
- Supports multiple gesture classes.
- Burst capture mode for fast data collection.
- Displays live camera preview with status information.


## How it works

1. Define your gesture labels in the `signharakat` list.
2. Run the script.
3. Press **Q** to start automatic image capture.
4. The script saves the specified number of images for the current class.
5. Repeat for every class.

## Configuration
```python
dir = "./dataset_collection"
signharakat = ["letA", "letB", "letC"]

sampnum = 100
delaybetweenimg = 0.15
```

You can change:

- dataset folder
- gesture names
- number of images
- capture delay

## Controls

| Key | Action |
|------|--------|
| Q | Start automatic image capture |
| ESC | Exit the program |

## Output

```
dataset_collection/
    letA/
    letB/
    letC/
```

Each folder contains captured images for one gesture class.
Creating Dataset (creating_dataset.py)
# Dataset Creator

This script converts collected hand images into numerical landmark data using MediaPipe Hand Landmarker.

## Features

- Detects up to two hands.
- Extracts 21 landmarks per hand.
- Stores x and y coordinates only.
- Automatically pads missing hands with zeros.
- Saves the processed dataset as a pickle file.

## Input

```
dataset_collection/
    Class_A/
    Class_B/
    ...
```

## Output

```
l_hands_points.pickle
```

The pickle file contains:

```python
{
    "data": [...],
    "label": [...]
}
```

Each sample contains:

- Right hand: 42 values
- Left hand: 42 values

Total:

```
84 features per sample
```

## Requirements

- MediaPipe Tasks
- hand_landmarker.task model
- Collected dataset images

## Notes

Images where no hand is detected are skipped automatically.





# Train Classifier

This script trains a Random Forest classifier using the generated landmark dataset.

## Features

- Loads landmark data from pickle.
- Splits dataset into training and testing sets.
- Uses stratified sampling when possible.
- Trains a Random Forest model.
- Calculates classification accuracy.
- Saves the trained model.

## Input

```
l_hands_points.pickle
```

## Output

```
fmodel.pickle
```

## Workflow

1. Load dataset.
2. Train/Test split.
3. Train Random Forest.
4. Evaluate accuracy.
5. Save trained model.

## Example Output(one of the tested outputs)

```
Total images: 600

Accuracy: 98.74%

Model successfully saved.
```

## Machine Learning Model

```
RandomForestClassifier
```

No parameter tuning is required for the default version.



# Live Gesture Recognition

This script performs real-time hand gesture recognition using the trained Random Forest model.

## Features

- Real-time webcam prediction.
- Detects up to two hands.
- Smart right/left hand ordering.
- Draws bounding boxes.
- Displays predicted gesture labels.

## Requirements

- fmodel.pickle
- hand_landmarker.task
- Webcam

## Workflow

1. Capture webcam frame.
2. Detect hand landmarks.
3. Build an 84-feature vector.
4. Predict gesture.
5. Display prediction on screen.

## Controls

| Key | Action |
|------|--------|
| Q | Quit |

## Display

The application shows:

- Live webcam feed
- Bounding box
- Predicted gesture label

## Notes

If only one hand is detected, the missing hand is automatically represented with zeros to keep the feature vector length fixed.4