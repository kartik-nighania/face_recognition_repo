# to use:
# python face_detection_and_recognition.py --shape-predictor shape_predictor_68_face_landmarks.dat --encodings encodings.pickle --dataset dataset --camera 0

'''
to do face detection from video feed.
show landmark points too on live video feed as well.

take screenshot and save it if:

single face is detected.
given face size to window ratio is sufficient.
eyes status with eye aspect ratio
face landmark detection to make sure face looks straight and front.
'''

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# face size aspect ratio to frame. Only save image if above threshold.
FACE_AR_THRESH = 0.25

# webcam resolution
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import pickle
import dlib
import cv2
import face_recognition
import sys
import os

# algorithms implemented in utilities.py file
from utilities import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-c", "--camera", required=True, type=int,
	help="camera 0 for laptop webcam. other number for external webcam.")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--dataset", required=True,
	help="path to images to paste on id card")
ap.add_argument("-p", "--pose-estimation", type = bool, default=False,
	help="do 3d pose estimation of the head and show a line on the head in 3D")
args = vars(ap.parse_args())
	
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
frame_height = 0
frame_width = 0
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")

vs = cv2.VideoCapture(args['camera'])
ret = vs.set(3, WEBCAM_WIDTH)
ret = vs.set(4, WEBCAM_HEIGHT)

time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# get frames
	_,frame = vs.read()
	#frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	frame_width = vs.get(3)
	frame_height = vs.get(4)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	if len(rects) == 1:
		for rect in rects:

			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			
			# average the eye aspect ratio together for both eyes
			ear = eye_aspect_ratio(shape)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1

			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1

				# reset the eye frame counter
				COUNTER = 0
			
			# finding the face aspect ratio
			far = face_aspect_ratio(shape, frame_width)

			# SAVE IMAGE IF CONDITIONS MET:
			'''
			SINGLE FACE : done
			ASPECT RATIO THRESHOLD
			EYES OPEN THRESHOLD
			POSE CENTER
			'''
			if(ear > EYE_AR_THRESH and far > FACE_AR_THRESH):

				# RECOGNIZE THE PERSON
				name = recognize_face(frame, data)
				print(name)
				# the name is the file name in dataset folder. 
				# take the image and resize it to passport size.
				if (name != None):
					
					path = os.path.join(args["dataset"], name)
					matched_image = cv2.imread(path)

					cv2.imshow("CELEB MATCH", matched_image)

			# DRAW ALL THE THINGS ON THE FRAME

			fullFaceConvex = face_hull(shape)
			(leftEyeHull, rightEyeHull) = eye_hulls(shape)

			if args["pose_estimation"]:
				print("doing 3D pose estimation")
				(nose_end_point2D, jacobian, head_3d_base, head_3d_end) = pose_estimation(shape, frame_width, frame_height)
				cv2.line(frame, head_3d_base, head_3d_end, (500,0,0), 2)

			cv2.drawContours(frame, [fullFaceConvex], -1, (255, 0, 0), 1)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			#uncheck to draw all facial landmark points
			#draw_face_landmarks(frame, shape)

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Eye status: " + ("OPEN" if ear > EYE_AR_THRESH else "CLOSED"), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Face pose estimation: TRACKING", (10, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Eye aspect ratio: {:.2f}".format(ear), (10, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Face to frame ratio: {:.2f}".format(far*100) +"%", (10, 110),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
	
	else:
		cv2.putText(frame, str(len(rects)) + " Faces detected. Only 1 allowed.", (90, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	if key == ord("a"):
		# wait till its pressed again
		while True:
			if (cv2.waitKey(1) & 0xFF) == ord("d"):
				break

 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()