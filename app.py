import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('gesture_model.h5')
gestures = ['thumbs_up', 'thumbs_down', 'ok_sign', 'gang_sign']

st.title("Real-Time Hand Gesture Recognition")

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

ret, frame = cap.read()
if ret:
    st.image(frame, channels='BGR')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_image = cv2.resize(frame, (64, 64))
            hand_image = np.expand_dims(hand_image, axis=0) / 255.0
            prediction = model.predict(hand_image)
            gesture = gestures[np.argmax(prediction)]
            
            st.write(f"Gesture: {gesture}")
    
    if st.button("Stop"):
        break

cap.release()
