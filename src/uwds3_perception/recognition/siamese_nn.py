from keras.models import Model
from keras.layers import Input, Dense, Activation


class SiameseNeuralNetwork(object):
    def __init__(self, input_model, output_model):
        """Creates the model"""
        self.input_model = input_model
        self.output_model = output_model

        input_1 = Input(shape=self.input_model)
        input_2 = Input(shape=self.input_model)

        output_1 = self.output_model(input_1)
        output_2 = self.output_model(input_2)

        output = self.output_model([output_1, output_2])
        self.model = Model([input_1, input_2], output)

    def load(self, checkpoint_path):
        """Load pretrained weights"""
        self.model.load_weights(checkpoint_path)

    def compile(self, *args, **kwargs):
        """Compile the siamese network"""
        self.model.compile(*args, **kwargs)

    def train():
        """Train the siamese network"""
        pass

    def generate_pairs(x_data, y_data, batch_size):
        """Generate the taining pairs (positive and negative)"""
        pass

    def predict(self):
        """ """
        return self.model
