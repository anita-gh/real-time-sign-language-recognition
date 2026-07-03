import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python
import pickle


# figure out where this script is located(not limiting it to one path(path handeling))
script_dir = os.path.dirname(os.path.realpath(__file__))
print("system path settings")


model_path = os.path.join(script_dir, 'hand_landmarker.task')
print(f"AI file path: {model_path}") #prints the path for needed debugging/for me


dataset_dir = os.path.normpath(os.path.join(script_dir, '..', 'dataset_collection'))
print(f"Your photos path: {dataset_dir}\n")




# check if there is any dataset at all (for me)
if not os.path.exists(dataset_dir):
    print(f"dataset folder was not found at {dataset_dir}!")
    exit()

# to check for  .task
print(os.path.exists(model_path))


# Set up the hand detector (max 2 hands)
base_options = mp_python.BaseOptions(model_asset_path=model_path) # Load the hand detection model
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) # Detect up to two hands
handsfinder = vision.HandLandmarker.create_from_options(options) # Create the detector


l_hands_points = []
barchasb = []


#for me
print("Dataset folder found")
print("Processing images...")


# going through the image folder for processing
for dir_ in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, dir_)

    if not os.path.isdir(class_path):
        continue
        
    succes_c = 0
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    for img_path in image_files:
        full_img_path = os.path.join(class_path, img_path)
        
        # uploading imges in mediapipe format
        mp_image = mp.Image.create_from_file(full_img_path)
        image_processing_options = vision.ImageProcessingOptions()
        
        # hand detection
        detection_result = handsfinder.detect(mp_image, image_processing_options)
        
        if detection_result.hand_landmarks:
            data_aux = []
            all_hands = detection_result.hand_landmarks
            
            # fisrt hand (it will be 42 land marks)
            for landmark in all_hands[0]:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
                
            # chaking for the second hand
            if len(all_hands) > 1:
                #if found add the cordinatoins (48 in total --> 42 for each hand)
                for landmark in all_hands[1]:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
            else:
                # If there were no second hand 42 zeros will be added to make the array length exactly 84 to not get errors
                data_aux.extend([0.0] * 42)
                
            
            if len(data_aux) == 84: # check the array to be exactly 84 to net get any errors and then add them
                l_hands_points.append(data_aux)
                barchasb.append(dir_)
                succes_c += 1

    #for me
    print(f"-> class {dir_}: {succes_c} / {total_images} images successfull")

# 84 landmarks both hands landmarks
pickle_output_path = os.path.join(script_dir, 'l_hands_points.pickle')

# save all the fingers and each class name data 
with open(pickle_output_path, 'wb') as f:
    pickle.dump({'data': l_hands_points,    
                 'label': barchasb}, f)

print(f"processing finished 2 hands datasets successfully saved at:\n{pickle_output_path}")