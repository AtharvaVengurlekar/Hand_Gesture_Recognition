# Hand Gesture Recognition

## Create Virstual Environment
You can use venv or conda environment as per your need.


```python -m venv /path/to/you/environment/```

## Installation
Run the [requirements.txt](/requirements.txt) file.
```pip install -r requirements.txt``` 
or 
Run "pip install tensorflow mediapipe opencv-python" command to install all the necessary libraries to run the project

## Data collection process
To collect the data, run ```python data_collection.py```. This will collect the hand data using the mediapipe library. Following signs are in correct pipeline:

* Thumbs up 👍
* Thumbs down 👎
* Ok sign ✌️
* Gang sign 🤟

## Data Proprocessing
To preprocess the data, run python ```data_preprocessing.py```.This script preprocesses the collected hand gesture images to prepare them for model training. The following steps are performed:

* Resizing images to a standard size.
* Normalization of pixel values for neural network input.
* Data augmentation to artificially increase the dataset size by  applying transformations like rotation and flipping.
* Label encoding for the hand gesture categories (e.g., thumbs up, thumbs down, etc.).

## Model Training
To train the model run ```python train_model.py```. This script trains a Convolutional Neural Network (CNN) model on the preprocessed hand gesture dataset. The model learns to classify the different hand gestures collected during the data collection process.

Once the training is completed, the model will be saved as a .h5 file for later use in gesture recognition.

## Gesture Recognition
To test the model run, ```python gesture_recognition.py```.This script is used for real-time hand gesture recognition. It loads the trained model from the .h5 file and uses the webcam to capture live gestures. The model predicts the gesture based on the live input and displays the recognized gesture on the screen.

Once executed, the webcam will open, and it will start recognizing hand gestures from the live feed.

## User Interface
Run the ```python app.py```.The app.py file is the main script that ties together the entire hand gesture recognition system and provides a user interface for the gesture recognition process.

Main Features:

* Flask Web Server: This file sets up a Flask web server for deploying the hand gesture recognition system as a web application.
* Real-Time Gesture Recognition: It integrates the trained hand gesture model with the web application and uses the webcam to capture gestures in real-time.
* Model Integration: The trained model.h5 is loaded, and predictions are made based on the gestures captured via the webcam.
* Web Interface: The web interface allows users to view the gesture recognition output, displaying the recognized gesture on the web page.









