import time
import os
import cv2


dir = './dataset_collection'
signharakat = ['letA', 'letB', 'letC']
sampnum = 100
delaybetweenimg = 0.15 # a little shorter than 0.20 so the cammera won't crash

# using defoult camera of laptop/ first camera of system (and cheking if there is any camera connected to system)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("couldn't find the camera please make sure you have camera conneted to your system")
    exit()

#makes the directory if it didnt exist
if not os.path.exists(dir):
    os.makedirs(dir)

for label in signharakat:
    label_path = os.path.join(dir, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print(f'>>> taking pictures for: {label}')
    
    imgcounter = 0
    takingimg = False

    while imgcounter < sampnum:
        success, frame = cam.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        show = frame.copy()

        # the text thtas on the screen
        if not takingimg:
            status_text = f'now will take photo for: {label} press Q for auto photo taking'
            text_color = (0, 165, 255) 
        else:
            status_text = f'auto photo taking fot {label} -> [{imgcounter}/{sampnum}]'
            text_color = (0, 255, 0) 

        cv2.putText(show, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        cv2.imshow('Interactive Data Logger', show)
        
        # waitkey is 30 here to make keyboard work smoother
        # but we make it 1 when recording so we won't have frame drop
        wait_time = 1 if takingimg else 30
        key_pressed = cv2.waitKey(wait_time) & 0xFF

        if key_pressed == ord('q') and not takingimg:
            takingimg = True
            print(f"Starting Auto-Burst for {label}")
            continue 

        elif key_pressed == 27: #esc
            print("Process terminated by user.")
            cam.release()
            cv2.destroyAllWindows()
            exit()

        # auto capture mode starts after user presses Q
        if takingimg:
            file_name = f'sample_{label}_{imgcounter}.jpg'
            finalpath = os.path.join(label_path, file_name)
            
            cv2.imwrite(finalpath, frame)
            print(f'auto saved: {file_name}')
            imgcounter += 1
            
            time.sleep(delaybetweenimg)

    # reseting everything for the nex class
    takingimg = False

cam.release()
cv2.destroyAllWindows()
print("data collection was successful")