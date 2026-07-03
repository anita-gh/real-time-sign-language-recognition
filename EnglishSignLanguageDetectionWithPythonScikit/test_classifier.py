import os
import pickle
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(dir, 'fmodel.pickle')
tskmodel_path = os.path.join(dir, 'hand_landmarker.task')

if not os.path.exists(pickle_path):
    print("fmodel.pickle was not found")
    exit()

dictmodel = pickle.load(open(pickle_path, 'rb'))
model = dictmodel['fmodel']

base_options = python.BaseOptions(model_asset_path=tskmodel_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
print("testing with Smart Hand Ordering")

while True:
    ret, frame = cap.read()
    if not ret: break

    
    height, width, _ = frame.shape
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    if_found = detector.detect(mp_image, vision.ImageProcessingOptions())

    if if_found.hand_landmarks:
        x_coords = []
        y_coords = []
        
        # making separate containers so if no second hand found it won't make problem
        right_hand_data = []
        left_hand_data = []

        # examining each hand found in the image
        for idx, hand_landmarks in enumerate(if_found.hand_landmarks):
            hand_label = if_found.handedness[idx][0].category_name # find the name of the hand (Left or Right)
            
            temp_data = []
            for landmark in hand_landmarks:
                temp_data.append(landmark.x)
                temp_data.append(landmark.y)
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            
            if hand_label == "Right":
                right_hand_data = temp_data
            else:
                left_hand_data = temp_data

        # Repeated assumption: first the right hand (42) and then the left hand (42)
        if not right_hand_data:
            right_hand_data = [0.0] * 42
        if not left_hand_data:
            left_hand_data = [0.0] * 42

        data_aux = right_hand_data + left_hand_data

        # box around the hand
        x1, y1 = max(0, int(min(x_coords) * width) - 20), max(0, int(min(y_coords) * height) - 20)
        x2, y2 = min(width, int(max(x_coords) * width) + 20), min(height, int(max(y_coords) * height) + 20)

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_gesture = prediction[0].replace('Class_', '')

            cv.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 3)
            cv.putText(frame, predicted_gesture, (x1, y1 - 15), cv.FONT_HERSHEY_DUPLEX, 1.5, (46, 204, 113), 2, cv.LINE_AA)

    cv.imshow('Sign Language Detection - Live', frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()