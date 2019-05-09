## TO install:

pip install -r requirements.txt

## TO train:

source face_recognition_app/bin/activate

python encode_faces.py --dataset dataset --encodings encodings.pickle

python face_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --encodings encodings.pickle --id-images dataset --camera 1

## BUTTONS
q on the image to close the app

a on the image to stop the app

d on the image to start it again
