Tested with Python 3.10.13 

# Emotion_detection_with_CNN

![emotion_detection](https://github.com/datamagic2020/Emotion_detection_with_CNN/blob/main/emoition_detection.png)

### Packages need to be installed

Make sure you are running `python 3.10.13`

- python3 -m venv .venv
- pip install -r requirement.txt

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TranEmotionDetector.py

It will take several hours depends on your processor. (On i7 processor with 16 GB RAM it took me around 4 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
`python TestEmotionDetector.py`

Sample video can be downloaded from https://www.pexels.com/video/three-girls-laughing-5273028/. Once downloaded rename it to emotion_sample.mp4 and move it to sample folder.