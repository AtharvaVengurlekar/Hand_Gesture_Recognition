import cv2
import os
import mediapipe as mp
import numpy as np

# Define the gestures
gestures = ['thumbs_up', 'thumbs_down', 'ok_sign', 'gang_sign']
num_images_per_gesture = 200  # Number of images per gesture

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Set up paths for storing data
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create folders for each gesture if they don't exist
for gesture in gestures:
    folder_path = os.path.join(data_dir, gesture)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

cap = cv2.VideoCapture(0)

print("\nStarting data collection. Perform each gesture sequentially.\n")
print("Press 'q' to stop collecting data at any time.")

for gesture in gestures:
    print(f"\nStarting collection for: {gesture}. Please perform the gesture...")

    img_count = 0
    while img_count < num_images_per_gesture:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save the frame for the current gesture
            folder_path = os.path.join(data_dir, gesture)
            img_name = os.path.join(folder_path, f"{gesture}_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved: {img_name}")
            img_count += 1

        # Display the frame
        cv2.imshow("Data Collection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\nData collection completed.")
cap.release()
cv2.destroyAllWindows()
