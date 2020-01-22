#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras import backend as K
import os
import scipy
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from tqdm import tqdm


def initialize_weights(shape, name=None):
    """Initialize weights"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def initialize_bias(shape, name=None):
    """Initialize bias"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


class OneShotCNNLearner(object):
    def __init__(self, input_shape, weights_path=None):
        self.training_model = self.create_training_model(input_shape)
        self.training_model.compile(loss="binary_crossentropy", optimizer=Adam(lr = 0.00006))
        if weights_path is not None:
            self.training_model.load_weights(weights_path)
            self.embedding_model = self.create_embedding_model(input_shape, self.training_model)
            self.embedding_model.compile(loss="binary_crossentropy", optimizer=Adam(lr = 0.00006))

    def create_training_model(self, input_shape):
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)

        base_model = Sequential()
        base_model.add(Conv2D(64, (10, 10),
                       activation='relu',
                       input_shape=input_shape,
                       kernel_initializer=initialize_weights,
                       kernel_regularizer=l2(2e-4)))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(128,
                       (7, 7),
                       activation='relu',
                       kernel_initializer=initialize_weights,
                       bias_initializer=initialize_bias,
                       kernel_regularizer=l2(2e-4)))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(128,
                              (4, 4),
                              activation='relu',
                              kernel_initializer=initialize_weights,
                              bias_initializer=initialize_bias,
                              kernel_regularizer=l2(2e-4)))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(256,
                              (4, 4),
                              activation='relu',
                              kernel_initializer=initialize_weights,
                              bias_initializer=initialize_bias,
                              kernel_regularizer=l2(2e-4)))

        base_model.add(Flatten())
        base_model.add(Dense(4096,
                             activation='sigmoid',
                             kernel_regularizer=l2(1e-3),
                             kernel_initializer=initialize_weights,
                             bias_initializer=initialize_bias))

        embedding_1 = base_model(input_1)
        embedding_2 = base_model(input_2)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([embedding_1, embedding_2])

        head_model = Dense(1,
                           activation='sigmoid',
                           bias_initializer=initialize_bias)(L1_distance)

        siamese_model = Model(inputs=[input_1, input_2], outputs=head_model)
        return siamese_model

    def create_embedding_model(object, input_shape, training_model):
        base_model = Sequential()
        base_model.add(Conv2D(64, (10, 10),
                       activation='relu',
                       input_shape=input_shape,
                       weights=training_model.layers[0].get_weights()))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(128,
                       (7, 7),
                       activation='relu',
                       weights=training_model.layers[1].get_weights()))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(128,
                              (4, 4),
                              activation='relu',
                              weights=training_model.layers[2].get_weights()))

        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(256,
                              (4, 4),
                              activation='relu',
                              weights=training_model.layers[3].get_weights()))

        base_model.add(Flatten())
        base_model.add(Dense(4096,
                             activation='sigmoid',
                             weights=training_model.layers[4].get_weights()))
        return base_model

    def draw_weights(self, mode="train"):
        raise NotImplementedError()

    def draw_activations(self, mode="train"):
        raise NotImplementedError()

    def extract(self, rgb_image):
        return self.embedding_model.predict(rgb_image)

    def predict(self, rgb_image_1, rgb_image_2):
        return self.training_model.predict(rgb_image_1, rgb_image_2)


class OneshotDataLoader(object):
    """
    Manage the dataset and oneshot test procedures for the given siamese network and dataset
    See : https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
    """
    def __init__(self, train_directory_path, val_directory_path):
        print("Start loading the dataset:\r\n'{}'\r\n'{}'".format(train_directory_path, val_directory_path))
        self.X_train, self.Y_train, self.train_classes = self.load_dataset(train_directory_path)
        self.X_val, self.Y_val, self.val_classes = self.load_dataset(val_directory_path)
        print("Training categories ({} different):".format(len(self.train_classes.keys())))
        print("{}\r\n".format(self.train_classes.keys()))
        print("Validation categories ({} different from training):".format(len(self.val_classes.keys())))
        print("{}\r\n".format(self.val_classes.keys()))

    def load_dataset(self, data_directory_path, n=0, verbose=True):
        """Load the dataset from given dir"""
        X_data = []
        Y_data = []
        individual_dict = {}
        category_dict = {}
        for c in os.listdir(data_directory_path):
            category_dict[c] = [n, 0]
            class_path = os.path.join(data_directory_path, c)
            for individual in os.listdir(class_path):
                individual_dict[n] = (c, individual)
                individual_images = []
                individual_path = os.path.join(class_path, individual)
                for filename in os.listdir(individual_path):
                    image_path = os.path.join(individual_path, filename)
                    image = scipy.misc.imread(image_path)
                    individual_images.append(image)
                    Y_data.append(n)
                try:
                    X_data.append(np.stack(individual_images))
                except ValueError as e:
                    print("Exception occured: {}".format(e))
                n += 1
                category_dict[c][1] = n - 1

        Y_data = np.vstack(Y_data)
        X_data = np.stack(X_data)
        return X_data, Y_data, category_dict

    def create_batch(self, batch_size, mode="train"):
        """
        Create the positive and negative pairs
        """
        if mode == "train":
            X = self.X_train
        else:
            X = self.X_val
        n_classes, n_examples, w, h = X.shape

        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        # Initialize the batch
        pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
        targets = np.zeros((batch_size,))

        targets[batch_size//2:] = 1

        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, n_examples)

            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has a different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes

            pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

        return pairs, targets

    def get_training_generator(self, batch_size):
        """
        Generates batches, so model.fit_generator can be used.
        """
        while True:
            pairs, targets = self.create_batch(batch_size, mode="train")
            yield (pairs, targets)

    def get_validation_generator(self, batch_size):
        """
        Generates batches, so model.fit_generator can be used.
        """
        while True:
            pairs, targets = self.create_batch(batch_size, mode="val")
            yield (pairs, targets)

    def train(self, model, *args, **kwargs):
        """
        Train the siamese network
        """
        print("Start training :")
        batch_size = kwargs.pop('batch_size')

        train_generator = self.get_training_generator(batch_size)
        val_generator = self.get_validation_generator(batch_size)

        train_steps = max(len(self.X_train) / batch_size, 1)
        val_steps = max(len(self.X_val) / batch_size, 1)

        model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            validation_data=val_generator,
                            validation_steps=val_steps, **kwargs)

    def make_oneshot_task(self, N_way, mode="val", category=None):
        """
        Creates pairs of test image, support set for testing N way one-shot learning.
        """
        if mode == 'train':
            X = self.X_train
            categories = self.train_classes
        else:
            X = self.X_val
            categories = self.val_classes
        n_classes, n_examples, w, h = X.shape

        indices = rng.randint(0, n_examples, size=(N_way,))
        if category is not None: # if language is specified, select characters for that language
            low, high = categories[category]
            if N > high - low:
                raise ValueError("This category ({}) has less than {} individual".format(category, N_way))
            categories = rng.choice(range(low, high), size=(N_way,), replace=False)
        else: # if no class specified just pick a bunch of random
            categories = rng.choice(range(n_classes), size=(N_way,), replace=False)

        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1, :, :]]*N_way).reshape(N_way, w, h, 1)
        support_set = X[categories, indices, :, :]
        support_set[0, :, :] = X[true_category, ex2]
        support_set = support_set.reshape(N_way, w, h, 1)
        targets = np.zeros((N_way,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]
        return pairs, targets

    def test_oneshot(self, model, N_way, trials, mode="val", verbose=True):
        """
        Tests average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
        """
        n_correct = 0
        if verbose:
            print("Evaluating model {} on {} random {} way one-shot learning tasks ...".format(trials, N_way))
        for i in range(trials):
            inputs, targets = self.make_oneshot_task(N_way, mode=mode)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / trials)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N_way))
        return percent_correct

    def nearest_neighbour_correct(self, pairs, targets):
        """
        returns 1 if nearest neighbour gets the correct answer for a one-shot task
            given by (pairs, targets)
        """
        L2_distances = np.zeros_like(targets)
        for i in range(len(targets)):
            L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
        if np.argmin(L2_distances) == np.argmax(targets):
            return 1
        return 0

    def test_nn_accuracy(self, N_way, trials, verbose=True):
        """
        Returns accuracy of NN approach
        """
        if verbose:
            print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(trials, N_way))
        n_right = 0

        for i in range(trials):
            pairs, targets = self.make_oneshot_task(N_way, "val")
            correct = self.nearest_neighbour_correct(pairs, targets)
            n_right += correct
        return 100.0 * n_right / trials

    def train_and_evaluate(self, model, weights_path="/tmp/", N_way=20, trials=250, epochs=20000, evaluate_every=200, batch_size=32):
        t_start = time.time()
        best = -1
        for i in tqdm(range(1, epochs+1)):
            inputs, targets = self.create_batch(batch_size)
            loss = model.train_on_batch(inputs, targets)
            if i % evaluate_every == 0:
                print("------ Evaluation at epoch {} ------".format(i))
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                print("Train Loss: {0}".format(loss))
                val_acc = self.test_oneshot(N_way, trials, mode="val", verbose=True)
                model.save_weights(os.path.join(weights_path, 'weights.{}.h5'.format(i)))
                if val_acc >= best:
                    print("Current best: {0}, previous best: {1}".format(val_acc, best))
                    best = val_acc
                print("Continue training...")

    def evaluate(self, model, model_name, N_way=20, trials=50, verbose=True):
        ways = np.arange(1, N_way, 2)
        val_accs, train_accs, nn_accs = [], [], []

        for N in ways:
            val_accs.append(self.test_oneshot(model, N, trials, mode="val", verbose=verbose))
            train_accs.append(self.test_oneshot(model, N, trials, mode="train", verbose=verbose))
            nn_accs.append(self.test_nn_accuracy(N, trials, verbose=verbose))

        plt.plot(ways, val_accs, "m", label=model_name+"(val)")
        plt.plot(ways, train_accs, "y", label=model_name+"(train)")
        plt.plot(ways, nn_accs, "b", label="Nearest neighbour")
        plt.plot(ways, 100.0/ways, "g", label="Random guessing")
        plt.xlabel("Number of possible classes in one-shot tasks")
        plt.ylabel("% Accuracy")
        plt.title("One-Shot Learning Performance")
        plt.legend(loc='center left')
        plt.show()


if __name__ == '__main__':
    train_dir = "../../../data/omniglot/omniglot/python/images_background/"
    val_dir = "../../../data/omniglot/omniglot/python/images_evaluation/"
    input_shape = (105, 105, 1)
    model = OneShotCNNLearner(input_shape).training_model
    data_loader = OneshotDataLoader(train_dir, val_dir)
    data_loader.train(model, batch_size=32, epochs=1)
    data_loader.evaluate(model, "OneShotCNNLearner")
