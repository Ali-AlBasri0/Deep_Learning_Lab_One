# Facial Emotion Recognition (FER2013)

A neural network that predicts 1 of 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise) from a face image, with a Streamlit app that uses the live webcam.

## Files
- `Emotion_Recognition.ipynb`: data exploration, preprocessing, models, evaluation, error analysis
- `app.py`: Streamlit webcam app (shows the predicted emotion and all 7 probabilities)
- `emotion_model.keras`: the trained model (created by the last cell of the notebook)
- `requirements.txt`: packages for the app

## Setup
1. Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and put it in a folder called `fer2013/` (with `train/` and `test/` inside)
2. Install the packages: `pip install -r requirements.txt` (the notebook also needs `scikit-learn` and `matplotlib`)
3. Run the notebook to train the model and create `emotion_model.keras`
4. Run the app: `streamlit run app.py`

## Data
- 48x48 grayscale face images, 7 classes
- 28,709 training images: 22,968 for training and 5,741 for validation (80/20 split)
- 7,178 test images (the `test` folder)
- Preprocessing: images are resized to 48x48 by `image_dataset_from_directory`, and pixels are normalized (0-255 to 0-1) with a `Rescaling` layer inside the model
- The classes are imbalanced: `happy` has 7,215 training images and `disgust` only 436

## Models
Both models use `sparse_categorical_crossentropy` and the `adam` optimizer.

| Model | Architecture | Training | Test accuracy |
|---|---|---|---|
| Baseline | Rescaling, Flatten, Dense 128 (relu), Dense 64 (relu), Dense 7 (softmax) | 5 epochs | 0.35 |
| Improved (final) | Same layers | Up to 30 epochs with EarlyStopping (patience 3, best weights restored) | 0.37 |

The improvement comes from training longer without overfitting: EarlyStopping stopped the training at epoch 8 and kept the weights from epoch 5, which had the lowest validation loss (1.63). The macro F1 score went from 0.22 to 0.25.

## Results (improved model, test set)
| Class | Precision | Recall |
|---|---|---|
| angry | 0.29 | 0.04 |
| disgust | 0.00 | 0.00 |
| fear | 0.25 | 0.03 |
| happy | 0.41 | 0.74 |
| neutral | 0.31 | 0.49 |
| sad | 0.36 | 0.21 |
| surprise | 0.41 | 0.52 |

Accuracy is 0.37 on 7,178 test images (random guessing would be about 0.14).

## Error analysis
- The model mostly predicts `happy`, `neutral` and `surprise`, and almost never predicts `angry`, `fear` or `disgust` (recall of 0.04, 0.03 and 0.00)
- `disgust` has very few training images, so the model never learns it
- Many images are hard even for a person: faces turned to the side, hands on the face, glasses, very dark or very bright light
- Some labels in FER2013 are wrong or unclear
- Dense layers flatten the image, so the model loses the shape of the face (eyes and mouth). This is the main limit of this model

## How the app works
1. Take a photo with the webcam
2. The image is converted to grayscale, center-cropped and resized to 48x48
3. The model gives the probability of each emotion and the app shows the highest one with a bar chart of all 7

For the best result keep your face in the center of the frame with good light.

## Ethics & Limitations
- Lighting, head angle, glasses, masks and low image quality reduce accuracy
- FER2013 is not balanced across age, skin tone, gender and culture, so results can be different for different people
- The accuracy is low (0.37), so the predictions are often wrong and should be treated as a fun demo
- A facial expression is not the same as a real feeling, and cultures show emotions differently
- The app does not detect the face, it only crops the center of the image
- Not for hiring, exams, surveillance or any decision about a person
