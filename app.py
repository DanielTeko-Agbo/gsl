import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration,WebRtcMode
import av
import pickle
import sklearn
import warnings
import threading

warnings.filterwarnings('ignore')

holistic_model = mp.solutions.holistic # Holistic model
drawing_util = mp.solutions.drawing_utils # Drawing utilities

def kp_detection(image, model):
    image = cv2.cv2tColor(image, cv2.COLOR_BGR2RGB) # Converting image to RGB from opencv2's default BGR.
    image.flags.writeable = False  # Setting image to not writable.
    result = model.process(image) # Detecting keypoints from image.
    image.flags.writeable = True   # Setting image to writable.
    image = cv2.cv2tColor(image, cv2.COLOR_RGB2BGR) # Converting image back to BGR from RGB.
    return image, result

def kp_drawing(image, results):
    # Draw face landmarks 
    drawing_util.draw_landmarks(image,
                                results.face_landmarks,
                                holistic_model.FACEMESH_CONTOURS,
                                drawing_util.DrawingSpec(color=(20,15,10), thickness=1, circle_radius=1), # Keypoint style
                                drawing_util.DrawingSpec(color=(255,255,100), thickness=1, circle_radius=1) # Keypoint connection style
                                )

    # Draw pose landmarks  
    drawing_util.draw_landmarks(image, 
                                results.pose_landmarks, 
                                holistic_model.POSE_CONNECTIONS,
                                drawing_util.DrawingSpec(color=(100,255,255), thickness=2, circle_radius=2),
                                drawing_util.DrawingSpec(color=(100,255,255), thickness=2, circle_radius=1)
                                )

    # Draw left hand landmarks
    drawing_util.draw_landmarks(image, 
                                results.left_hand_landmarks, 
                                holistic_model.HAND_CONNECTIONS,
                                drawing_util.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2),
                                drawing_util.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2)
                                )
    
    # Draw right hand landmarks 
    drawing_util.draw_landmarks(image, 
                                results.right_hand_landmarks, 
                                holistic_model.HAND_CONNECTIONS,
                                drawing_util.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                drawing_util.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                                )


def extract_keypoints(result):
    """ A function to extract keypoints. """

    if result.right_hand_landmarks: # If the right hand was captured.
        # Assign to 'rh' the x,y and z keypoints of each landmark and append it to an array which is then flatttened into a 1-dimensional array. 
        rh = np.array([[mark.x, mark.y, mark.z] for mark in result.right_hand_landmarks.landmark]).flatten() 
    else:
        # Assign to 'rh' an array of zeros. 
        # rh has the same size in both instances.
        rh = np.zeros(63)

      # Extracting keypoints from the left hand landmarks
    if result.left_hand_landmarks: # If the left hand was captured.
        # Assign to 'lh' the x,y and z keypoints of each landmark and append it to an array which is then flatttened into a 1-dimensional array. 
        lh = np.array([[mark.x, mark.y, mark.z] for mark in result.left_hand_landmarks.landmark]).flatten() 
    else:
        # Assign to 'lh' an array of zeros. 
        # lh has the same size in both instances.
        lh = np.zeros(63)

    return np.concatenate([lh, rh])      

class OpencvVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_confidence = 0.5
        self.tracking_confidence = 0.5
        self.model = pickle.load(open('./model.pkl', 'rb'))

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
     
        # Set mediapipe model 
        with holistic_model.Holistic(min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.tracking_confidence) as holistic:
            while True:
                #img = frame.to_ndarray(format="bgr24")
                flip_img = cv2.flip(img,1)

                # Make detections
                image, results = kp_detection(flip_img, holistic)

                # Draw landmarks
                kp_drawing(image, results)

                # Extract keypoints 
                keypoints = extract_keypoints(results)
                
                # Make predictions
                prediction = self.model.predict([keypoints])

                # Displaying predictions
                cv2.rectangle(image, (0,2), (60,40), (117,117,117), -1)
                cv2.putText(image, str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(image,format="bgr24")

stream = webrtc_streamer(
    key="opencv-filter",
    video_processor_factory=OpencvVideoProcessor,
)

if stream.video_processor:
    stream.video_processor.detection_confidence = st.sidebar.slider("Detection Confidence", 0.00, 1.00, 0.50, 0.01)
    stream.video_processor.tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.00, 1.00, 0.50, 0.01)


