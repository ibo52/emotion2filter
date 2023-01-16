# emotion2filter
## Program detects emotion and applies simple filters to face by emotion.

- Place datasets to required directories (which explained text file in `dataset/` directory)
1. Run `landmark-model.py` to train model to detect facial landmarks
2. Run `emotion-model.py` to train model to detect facial emotions.
3. Run `run_camera.py` to see program outputs. __FPS is low__ due to CNN performances. I neither have a GPU nor implementation of GPU
- Program detects faces with openCV's haar cascade.(Frontal faces only) ![reference:](https://github.com/opencv/opencv/tree/4.x/data/haarcascades)



#Datasets references
DATASET | train data | test data|
--- | --- | --- |
FACIAL LANDMARK | train: http://users.sussex.ac.uk/~is321/training_images.npz | test: http://users.sussex.ac.uk/~is321/test_images.npz
EMOTION | https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset| test also included in link

# Sample outputs
## video -> ![link](https://github.com/ibo52/emotion2filter/blob/master/sample%20outputs/video1.webm)

##filter result of neutral face
![neutral face](https://github.com/ibo52/emotion2filter/blob/master/sample%20outputs/neutral.png)

## filter result of sad face
![link](https://github.com/ibo52/emotion2filter/blob/master/sample%20outputs/sad.png)

## landmark train results.
![landmark train result](https://github.com/ibo52/emotion2filter/blob/master/sample%20outputs/landmark-results.png)
