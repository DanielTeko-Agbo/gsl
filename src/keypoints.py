import mediapipe as mp
import cv2 as cv
import numpy as np

man = "hey There"
holistic_model = mp.solutions.holistic  # Holistic model to determine keypoints.
drawing_util = mp.solutions.drawing_utils  # Drawing utilities to draw keypoints.


def kp_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Converting image to RGB from opencv's default BGR.
    image.flags.writeable = False  # Setting image to not writable.
    result = model.process(image) # Detecting keypoints from image.
    image.flags.writeable = True   # Setting image to writable.
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # Converting image back to BGR from RGB.
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


# Extracting KeyPoints 
def extract_keypoints(results):
    """ A function to extract keypoints. """

    if results.right_hand_landmarks: # If the right hand was captured.
        # Assign to 'rh' the x,y and z keypoints of each landmark and append it to an array which is then flatttened into a 1-dimensional array. 
        rh = np.array([[mark.x, mark.y, mark.z] for mark in results.right_hand_landmarks.landmark]).flatten() 
    else:
        # Assign to 'rh' an array of zeros. 
        # rh has the same size in both instances.
        rh = np.zeros(63)


    # Extracting keypoints from the left hand landmarks
    if results.left_hand_landmarks: # If the left hand was captured.
        # Assign to 'lh' the x,y and z keypoints of each landmark and append it to an array which is then flatttened into a 1-dimensional array. 
        lh = np.array([[mark.x, mark.y, mark.z] for mark in results.left_hand_landmarks.landmark]).flatten() 
    else:
        # Assign to 'lh' an array of zeros. 
        # lh has the same size in both instances.
        lh = np.zeros(63)

    return np.concatenate([lh, rh])
