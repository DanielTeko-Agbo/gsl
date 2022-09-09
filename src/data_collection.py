import mediapipe as mp
import cv2 as cv
import csv
import numpy as np
import datetime

from keypoints import holistic_model, kp_detection, kp_drawing, extract_keypoints, man

# Data collection function
def data_collection(signs, img_num):
    capture = cv.VideoCapture(0)

    f = open(f'data/{datetime.datetime.now()}.csv', mode='w', newline='')
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    with holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sign in signs:
            for img in range(img_num):
            
                ret, frame = capture.read() 

                # Make keypoint detection.
                image, result = kp_detection(frame, holistic)

                # Draw landmarks 
                kp_drawing(image, result)

                # Show frame in a window.
                cv.imshow("OpenCV Feed", image) 

                # Applying Wait Logic
                if img == 0:
                    cv.putText(image, "STARTING COLLECTION", (120,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv.LINE_AA)
                    cv.putText(image, "Collecting frames for sign {}.  Image Number {}".format(sign, img), (12,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
                    cv.imshow('OpenCV Feed', image) # Show to screen
                    cv.waitKey(5000)
                else:
                    cv.putText(image, "Collecting frames for sign {}.  Image Number {}".format(sign, img), (12,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
                    cv.imshow('OpenCV Feed', image) # Show to screen
                    cv.waitKey(3000)

                # Saving Keypoints in a csv file.
                keypoints = extract_keypoints(result)
                csv_writer.writerow(np.append(keypoints, sign))
                

                # If there is a keyup and the pressed key is the 'spacebar', break out of the loop.
                if cv.waitKey(10) & 0xFF == ord(' '): 
                    break

        capture.release() # Close the video capture object 
        cv.destroyAllWindows() # Close all OpenCV windows.
    f.close()

data_collection(list('123'), 3)
