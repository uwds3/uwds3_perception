import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean, cosine
from ..detection.opencv_dnn_detector import OpenCVDNNDetector
from ..estimation.facial_features_estimator import FacialFeaturesEstimator


class OpenFaceRecognition(object):
    def __init__(self,
                 input_shape,
                 detector_model_filename,
                 detector_weights_filename,
                 frontalize=False,
                 metric_distance="euclidean"):
        raise NotImplementedError()
        # self.detector =
        # self.embedder =

    def extract(self, rgb_image):
        raise NotImplementedError()

    def predict(self, rgb_image_1, rgb_image_2):
        raise NotImplementedError()


class FacialRecognitionDataLoader(object):
    def __init__(self, train_directory_path, val_directory_path):
        raise NotImplementedError()

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
            inputs, targets = self.make_oneshot_task(N_way, mode=mode)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / trials)
        if verbose:
            print("Got an average of {}% {} way recognition accuracy".format(percent_correct, N_way))
        return percent_correct

    def make_recognition_task(self, N_way, mode="val", person=None):
