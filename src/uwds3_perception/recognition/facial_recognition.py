import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean, cosine
from ..detection.opencv_dnn_detector import OpenCVDNNDetector
from ..estimation.facial_features_estimator import FacialFeaturesEstimator
from ..detection.face_detector import FaceDetector
from pyuwds3.types.features import Features

class OpenFaceRecognition(object):
    def __init__(self,
                 input_shape,
                 detector_model_filename,
                 detector_weights_filename,
                 frontalize=False,
                 metric_distance="euclidean"):
        self.face_detector = FaceDetector()
        self.detector_model_filename = detector_model_filename
        self.facial_features_estimator = FacialFeaturesEstimator( detector_model_filename,detector_weights_filename)
        self.input_shape = input_shape
        self.detector_model_filename = detector_model_filename
        self.detector_weights_filename = detector_weights_filename
        self.frontalize = frontalize
        self.metric_distance = metric_distance

    def extract(self, rgb_image):
        face_list = self.detector.detect(rgb_image)
        if len(img_list) == 0:
            print("no image found for extraction")
        else:
            self.facial_features_estimator(rgb_image,face_list[0],frontalize)
            name = self.facial_features_estimator.name
            return face_list[0].features[name]


    def predict(self, rgb_image_1, rgb_image_2):
        feature1 = self.extract(rgb_image_1)
        feature2 = self.extract(rgb_image_2)
        return(self.metric_distance(feature1.to_array(),
                                    feature2.to_array() ))



class FacialRecognitionDataLoader(object):
    def __init__(self, train_directory_path, val_directory_path):
        #raise NotImplementedError()
        return

    def load_dataset(self, data_directory_path, n=0):
        X_data = []
        Y_data = []
        individual_dict = {}

        for c in os.listdir(data_directory_path):
            individual_dict[n] = c
            individual_path = os.path.join(data_directory_path, c)
            individual_images = []
            for snapshot_file in os.listdir(individual_path):
                image_path = os.path.join(individual_path, snapshot_file)
                image = cv2.imread(image_path)
                individual_images.append(image)
                Y_data.append(n)
            try:
                X_data.append(np.stack(individual_images))
            except ValueError as e:
                print("Exception occured: {}".format(e))
            n += 1
        X_data = np.stack(X_data)
        Y_data = np.vstack(Y_data)
        print(X_data.shape)
        print(Y_data.shape)
        return X_data, Y_data, individual_dict

    def test_recognition(self, model, N_way, trials, mode="val", verbose=True):
        """
        Tests average N way recognition accuracy of the embedding net over k trials
        """
        n_correct = 0
        if verbose:
            print("Evaluating model {} on {} random {} way recognition tasks...".format(trials, N_way))
        for i in range(trials):
            true_person,support_set, targets = self.make_oneshot_task(N_way, mode=mode)
            probs = []
            for i in support_set:
                probs.append(model.predict(true_person,support_test))
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / trials)
        if verbose:
            print("Got an average of {}% {} way recognition accuracy".format(percent_correct, N_way))
        return percent_correct

    def make_recognition_task(self, N_way, mode="val", person=None):
        """
        Creates pairs of test image, support set for testing N way learning.
        """
        if mode == 'train':
            X = self.X_train
            persons = self.train_classes
        else:
            X = self.X_val
            persons = self.val_classes

        n_classes, n_examples, w, h = X.shape
        if category is not None: # if person is specified,
            true_person = person
        else: # if no class specified just pick a bunch of random
            true_person = np.random.randint(n_classes)

        ex1, ex2 = rng.choice(X[true_person], replace=False, size=(2,))
        indices = rng.choice((range(true_person) + range(true_person+1, n_examples)),N_way-1)
        support_set = [ex2]
        for i in indices:
            support_set.append(rng.choice(X[i]))
        targets = np.zeros((N_way,))
        targets[0] = 1
        targets, support_set = shuffle(targets, support_set)

        return ex1,support_set, targets
