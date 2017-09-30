import cv2
import numpy as np
from keras.models import load_model

bg = None

def run_avg(image, acc_weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, acc_weight)

def segment(image, threshold=10):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.GaussianBlur(thresholded,(5,5),0)
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    model = load_model("model.h5")
    # initialize accumulated weight
    accumWeight = 0.5

    im_count = 0

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    x, y, r = 500, 900, 200
    # region of interest (ROI) coordinates
    top, right, bottom, left = x-r, y-r, x+r, y+r

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
            	print "[STATUS] please wait! calibrating..."
            elif num_frames == 29:
				print "[STATUS] calibration successfull..."
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region

                (thresholded, segmented) = hand
                epsilon = 0.01*cv2.arcLength(segmented,True)
                segmented = cv2.approxPolyDP(segmented,epsilon,True)

                # draw the segmented region and display the frame
                convex_hull = cv2.convexHull(segmented)

                cv2.rectangle(clone, (left, top), (right, bottom), (0,0,0), thickness=cv2.cv.CV_FILLED)
                cv2.drawContours(clone, [convex_hull + (right, top)], -1, (255, 0, 0), thickness=cv2.cv.CV_FILLED)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 255, 255), thickness=cv2.cv.CV_FILLED)

                preds = model.predict(cv2.resize(clone[top:bottom, right:left], (64, 64)).reshape((-1, 64, 64, 3)))[0]
                index = np.argmax(preds)
                text = ["rock", "paper", "scissors"][index] + " " + str(round(preds[index], 2))
                cv2.putText(clone, text, (right, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        path = None
        if keypress == ord("r"):
            path = "r" + str(im_count) + ".png"
        elif keypress == ord("p"):
            path = "p" + str(im_count) + ".png"
        elif keypress == ord("s"):
            path = "s" + str(im_count) + ".png"

        if path is not None:
            cv2.imwrite("data/" + path, clone[top:bottom, right:left])
            print "saved", path
            im_count += 1

    # free up memory
    camera.release()
    cv2.destroyAllWindows()
