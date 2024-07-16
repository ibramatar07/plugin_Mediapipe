import pickle, codecs
import mxw, mxw_imgui
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

class GestureControl:
    capture_device = -1  #default
    fist_detected = False
    last_detection_frame = 0
    camera_window_name = "Camera Feed"
    cap = None

instance_storage = {}

def onCreate():
    c = GestureControl()
    c.capture_device = get_valid_device()
    instance_storage[item_id] = c

    # OpenCV window to display camera feed
    cv2.namedWindow(c.camera_window_name)
    return

def get_valid_device():
    for device_index in range(10): 
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            if not is_ndi_camera(device_index):
                cap.release()
                return device_index
            cap.release()
    return -1


def onSave():
    serialized = codecs.encode(pickle.dumps(instance_storage[item_id]), "base64").decode()
    return serialized

def onLoad(serialized):
    instance_storage[item_id] = pickle.loads(codecs.decode(serialized.encode(), "base64"))
    return

def is_fist(landmarks):
    fist = True
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        if landmarks.landmark[finger_tip].y < landmarks.landmark[finger_pip].y:
            fist = False
            break
    return fist

def check_for_fist(results):
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            handedness = results.multi_handedness[results.multi_hand_landmarks.index(landmarks)].classification[0].label
            if handedness == "Right":
                if is_fist(landmarks):
                    return True
    return False

def onNewFrameInPlayoutCue():
    me = instance_storage[item_id]

    if me.capture_device == -1:
        print("No valid camera device found.")
        return

    if me.cap is None or not me.cap.isOpened():
        print(f"Initializing camera: {me.capture_device}")
        if me.cap is not None:
            me.cap.release()
        me.cap = cv2.VideoCapture(me.capture_device)

    if me.cap.isOpened():
        ret, img = me.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if check_for_fist(results):
                if not me.fist_detected:
                    me.fist_detected = True
                    me.last_detection_frame = mxw.framecounter
                    mxw.playlist.play()  # Advance to the next video
            else:
                me.fist_detected = False

            # Display camera feed in OpenCV window
            cv2.imshow(me.camera_window_name, img)
        else:
            print("Failed to read from camera.")
    else:
        print("Camera not opened.")

    cv2.waitKey(1)

def onRenderPanel():
    mxw_imgui.text_unformatted("This plugin detects a right hand fist gesture and advances to the next video.")
    me = instance_storage[item_id]
    
    if me.capture_device == -1:
        mxw_imgui.text_unformatted("No valid camera device found.")
    else:
        mxw_imgui.text_unformatted(f"Using camera device index: {me.capture_device}")

    return

def onClose():
    # Close the OpenCV window and release the camera when the plugin is closed
    me = instance_storage[item_id]
    if me.cap is not None:
        me.cap.release()
    cv2.destroyWindow(me.camera_window_name)
    return
