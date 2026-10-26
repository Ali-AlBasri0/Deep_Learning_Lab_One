import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model

st.title('Facial Emotion Recognition')

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model = load_model('emotion_model.keras')

photo = st.camera_input('Take a photo (face in the center)')

if photo is not None:
    img = Image.open(photo).convert('L')       # grayscale
    img = ImageOps.fit(img, (48, 48))          # center crop + resize to 48x48

    probs = model.predict(np.array(img).reshape(1, 48, 48, 1), verbose=0)[0]

    st.subheader('Prediction: ' + class_names[probs.argmax()])
    st.bar_chart(pd.Series(probs, index=class_names))
