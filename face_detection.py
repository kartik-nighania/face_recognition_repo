# to use:
# python face_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --encodings encodings.pickle --id-images dataset --camera 1

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

# ID card image location
ID_FACE_TL = (102, 364)
ID_FACE_BR = (377, 639)

# ID name location
ID_NAME_TL = (82, 659)
ID_NAME_BR = (383, 722)

# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-c", "--camera", required=True, type=int,
	help="camera 0 for laptop webcam. other number for external webcam.")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--id-images", required=True,
	help="path to images to paste on id card")
args = vars(ap.parse_args())

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def recognize_face(frame):
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model="cnn")
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			return max(counts, key=counts.get)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
def create_passport_pic(image_name):
	#image path
	path = os.path.join(args["id_images"], image_name)
	
	# read image, cut face and show it
	passport = cv2.imread(path)
	box = detector(passport, 0)

	# gives tuple in (x, y, w, h) format
	(x, y, w, h) = imutils.face_utils.rect_to_bb(box[0])
	
	#final_image = passport[x:(x+h+200), y:(y+w+200)]
	#passport = imutils.resize(passport, width=ID_FACE_BR[0] - ID_FACE_TL[0], height=ID_FACE_BR[1] - ID_FACE_TL[1])
	passport = imutils.resize(passport, height=ID_FACE_BR[1] - ID_FACE_TL[1])

	return passport
	
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()

vs = cv2.VideoCapture(args['camera'])
ret = vs.set(3, 640)
ret = vs.set(4, 480)
# ret = vs.set(3, 1280)
# ret = vs.set(4, 720)
time.sleep(2.0)

previous_name = ''

# loop over the frames from the video stream
while True:
	# get frames
	_,frame = vs.read()
	#frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	if len(rects) == 1:
		for rect in rects:
			#print("Rect " + str(rect) + " rect " + str(rect.tl_corner().x) + str(type(rect.tl_corner())))

			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			fullFaceConvex = cv2.convexHull(shape[0:27])

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

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
			
			# finding the face aspect rati0
			#height = vs.get(4)
			face_width = (dist.euclidean(shape[0], shape[16]) + dist.euclidean(shape[1], shape[15]))/2
			frame_width = vs.get(3)
			far = face_width/frame_width

			# Finding the faces POSE ESTIMATION
			     
			#2D image points. If you change the image, you need to change vector
			image_points = np.array([
										shape[33],     # Nose tip
										shape[8],      # Chin
										shape[36],     # Left eye left corner
										shape[45],     # Right eye right corner
										shape[48],     # Left Mouth corner
										shape[54]      # Right mouth corner
									], dtype="double")
			
			# 3D model points.
			model_points = np.array([
										(0.0, 0.0, 0.0),             # Nose tip
										(0.0, -330.0, -65.0),        # Chin
										(-225.0, 170.0, -135.0),     # Left eye left corner
										(225.0, 170.0, -135.0),      # Right eye right corne
										(-150.0, -150.0, -125.0),    # Left Mouth corner
										(150.0, -150.0, -125.0)      # Right mouth corner
									
									])
			
			
			# Camera internals
			focal_length = vs.get(3) # width
			center = (vs.get(3)/2, vs.get(4)/2)
			camera_matrix = np.array(
									[[focal_length, 0, center[0]],
									[0, focal_length, center[1]],
									[0, 0, 1]], dtype = "double"
									)

			 # Assuming no lens distortion
			dist_coeffs = np.zeros((4,1))

			(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
			
			# Project a 3D point (0, 0, 200) onto the image plane.
			# We use this to draw a line sticking out of the nose
			(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
			
			# SAVE IMAGE IF CONDITIONS MET:
			'''
			SINGLE FACE : done
			ASPECT RATIO THRESHOLD
			EYES OPEN THRESHOLD
			POSE CENTER
			'''
			if(ear > EYE_AR_THRESH and far > FACE_AR_THRESH):

				# RECOGNIZE THE PERSON
				name = recognize_face(frame)
				print(name)
				# the name is the file name in dataset folder. 
				# take the image and resize it to passport size.
				if (name != None):
					passport_pic = create_passport_pic(name)
					
					ID_card_image = cv2.imread("TECH_COALITION_2019.png")
					# print(str(ID_FACE_TL[0])+ " " + str(ID_FACE_TL[0]+passport_pic.shape[0]))
					# print(str(ID_FACE_TL[1])+ " " + str(ID_FACE_TL[1]+passport_pic.shape[1]))
					# print(passport_pic.shape)
					# print("id card: "+str(ID_card_image.shape))

					# REMOVE THE TEMP PICTURE AND FONT FRAME FROM TEMPLATE
					cv2.rectangle(ID_card_image, ID_FACE_TL, ID_FACE_BR, (255, 255, 255), cv2.FILLED)
					cv2.rectangle(ID_card_image, ID_NAME_TL, ID_NAME_BR, (255, 255, 255), cv2.FILLED)

					# put the face and name on the id card
					ID_card_image[ID_FACE_TL[1]:(ID_FACE_TL[1] + passport_pic.shape[0]), ID_FACE_TL[0]:(ID_FACE_TL[0]+passport_pic.shape[1])] = passport_pic
					cv2.putText(ID_card_image, name.split('.')[0], (ID_NAME_TL[0] + 10, ID_NAME_BR[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					

					# user_image = frame[rect.tl_corner().x:rect.tr_corner().x, rect.tl_corner().y:rect.bl_corner().y]
					# user_image.resize()

					cv2.imshow("ID_CARD", ID_card_image)
				else:
					ID_card_image = cv2.imread("TECH_COALITION_2019.png")
					cv2.imshow("ID_CARD", ID_card_image)

				cv2.putText(frame, "SAVED IMAGE", (10, 300),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# DRAW ALL THE THINGS ON THE FRAME

			cv2.drawContours(frame, [fullFaceConvex], -1, (255, 0, 0), 1)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# draw the line on the face nose
			p1 = ( int(image_points[0][0]), int(image_points[0][1]) - int(dist.euclidean(shape[21], shape[33])))
			p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]) - int(dist.euclidean(shape[21], shape[33])))
			cv2.line(frame, p1, p2, (500,0,0), 2)
			

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
	

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	
	else:
		cv2.putText(frame, str(len(rects)) + " Faces detected. Only 1 allowed.", (90, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		ID_card_image = cv2.imread("TECH_COALITION_2019.png")
		cv2.imshow("ID_CARD", ID_card_image)
	  
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