to install stable working opencv use:
pip install opencv-python

was able to install cuda toolkit 10.1 outside and then
renamed the folder in usr/local/cuda-10.1 to cuda

installed then dlib without seeing installing without cuda error



for dlib with gpu support:

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA




scripts to use:

python encode_faces.py --dataset dataset --encodings encodings.pickle
python recognize_faces_video.py --encodings encodings.pickle
python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png


mention installing dlib gpu if they have it. 
give pip uninstall dlib
then the readme file dlib code present in other folder


write code to add the shape predictor file with wget create sh script maybe


mention thresold
mention all the things you are calculating also with flag if possible
arrange the code better (create a helper file)

