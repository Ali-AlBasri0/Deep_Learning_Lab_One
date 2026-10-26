# Facial Emotion Recognition (FER2013)

A neural network that predicts 1 of 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise) from a face image, with a Streamlit app that uses the **live webcam**.

**Live app:** <paste your app link here>

## Files
- `FER_Emotion_Recognition.ipynb`: data exploration, preprocessing, models, evaluation, error analysis
- `app.py`: Streamlit webcam app (shows the predicted emotion and all 7 probabilities)
- `emotion_model.keras`: trained model (created by the last cells of the notebook)
- `requirements.txt`

## Setup
1. Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and put it in a folder called `fer2013/` (with `train/` and `test/` inside)
2. Install: `pip install -r requirements.txt` (the notebook also needs `scikit-learn` and `matplotlib`)
3. Run the notebook to train the model and create `emotion_model.keras`
4. Run the app: `streamlit run app.py`

## Data
- 48x48 grayscale face images, 7 classes
- Split: 80% train / 20% validation (from the `train` folder), the `test` folder is the test set
- Preprocessing: pixels normalized inside the model with a `Rescaling` layer
- Classes are imbalanced: `happy` has the most images (7215), `disgust` the fewest (436)

## Models
| Model | Architecture | Test accuracy |
|---|---|---|
| Baseline | Rescaling, Flatten, Dense 128, Dense 64, softmax 7 | 0.37 |
| Improved (final) | Rescaling (-1 to 1), Flatten, Dense 128 + Dropout 0.2, Dense 64 + Dropout 0.2, softmax 7 | _fill in_ |

- Loss: `sparse_categorical_crossentropy`, optimizer: `adam`
- Improvements tried: Dropout, EarlyStopping, lower learning rate (0.0005), pixels centered around 0, longer training
- Precision, recall per class and the confusion matrix are in the notebook

## How the app works
1. Take a photo with the webcam
2. The image is converted to grayscale, center-cropped and resized to 48x48
3. The model predicts the probability of each emotion and the app shows the highest one

For the best result keep your face in the center of the frame with good light.

## Ethics & Limitations
- Lighting, head angle, glasses, masks and low image quality reduce accuracy
- FER2013 is not balanced across age, skin tone, gender and culture, so results can be different for different people
- Very few `disgust` images, so the model almost never predicts it
- Labels are noisy (made by people online), so accuracy stays far from 100%
- A facial expression is not the same as a real feeling, and cultures show emotions differently
- The app does not find the face, it only crops the center of the image
- Not for hiring, exams, surveillance or any decision about a person
