import streamlit as st
import joblib
import os

model = joblib.load("emotion_model_logistic.pkl")

emotion_labels = {
    0: "Sad",
    1: "Happy",
    2: "Love",
    3: "Angry",
    4: "Fear"
}

gif_paths = {
    "Sad": "image/Sad.gif",
    "Happy": "image/Happy.gif",
    "Love": "image/Love.gif",
    "Angry": "image/Anger.gif",
    "Fear": "image/fear.gif"
}

st.title("Emotion Detection from Text")
st.write("Enter a sentence and the model will detect the emotion.")

user_input = st.text_area("Enter your text:", "")

if st.button("Analyze Emotion"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        emotion = emotion_labels.get(prediction, "Unknown Emotion")
        st.write(f"### Emotion Detected: **{emotion}**")

        gif_path = gif_paths.get(emotion)
        if gif_path and os.path.exists(gif_path):
            st.image(gif_path)
        else:
            st.warning("GIF not found for this emotion.")
    else:
        st.warning("Please enter some text to analyze.")
