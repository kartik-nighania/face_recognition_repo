# import the necessary packages
import face_recognition
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
import sys
import os

	
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def recognize_face(frame, data, rect):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb = imutils.resize(frame, width=750)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face

    # box by face_recognition is replaced by dlib box found and
    # converted in list format of face_recognition api by the function 
    #boxes = face_recognition.face_locations(rgb, model="cnn")
    boxes = rect_to_face_recog(rect)
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

def eye_aspect_ratio(shape):

    # extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    ear = 0
    for eye in (leftEye, rightEye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear += (A + B) / (2.0 * C)

	# return the average eye aspect ratio
    return ear/2

# Finding the faces POSE ESTIMATION
def pose_estimation(shape, width, height):            
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
    focal_length = width
    center = (width/2, height/2)
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

    head_3d_base = ( int(image_points[0][0]), int(image_points[0][1]) - int(dist.euclidean(shape[21], shape[33])))
    head_3d_end = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]) - int(dist.euclidean(shape[21], shape[33])))
    
    return (nose_end_point2D, jacobian, head_3d_base, head_3d_end)

# finding the face aspect ratio
def face_aspect_ratio(shape, width):
    face_width = (dist.euclidean(shape[0], shape[16]) + dist.euclidean(shape[1], shape[15]))/2
    far = face_width/width
    return far


def eye_hulls(shape):
    
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    
    # compute the convex hull for the left and right eye,
    # for visualization
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    
    return (leftEyeHull, rightEyeHull)

def face_hull(shape):
    face_points = shape[0:27]
    return cv2.convexHull(face_points)

def draw_face_landmarks(frame, shape):
    # loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


def rect_to_face_recog(rect):
	rect_face_list = [rect.tl_corner().y, rect.br_corner().x, rect.br_corner().y, rect.tl_corner().x]
	rect_face_list = [point if point > 0 else 0 for point in rect_face_list]
	return [tuple(rect_face_list)]