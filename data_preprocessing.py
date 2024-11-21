import os
import cv2
import numpy as np

IMG_SIZE = 64

def load_data():
    data = []
    labels = []
    gestures = ['thumbs_up', 'thumbs_down', 'ok_sign', 'gang_sign']
    
    for gesture in gestures:
        folder = f'data/{gesture}'
        
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' not found. Skipping...")
            continue
        
        label = gestures.index(gesture)
        
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
    
    return np.array(data), np.array(labels)

# Load data and save it as numpy files
data, labels = load_data()
if data.size > 0 and labels.size > 0:
    np.save('data.npy', data)
    np.save('labels.npy', labels)
else:
    print("No data found. Please ensure you have collected images for each gesture.")
