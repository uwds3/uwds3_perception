mkdir -p models/detection
cd models/detection
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb ssd_mobilenet_v2_coco_2018_03_29.pb
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt

rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz

wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt
wget https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb

cd -
mkdir -p models/estimation
cd models/estimation
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
rm shape_predictor_68_face_landmarks.dat.bz2

cd -
mkdir models/features
cd models/features
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7

wget https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/person-reidentification-retail-0076/person-reidentification-retail-0076.prototxt
