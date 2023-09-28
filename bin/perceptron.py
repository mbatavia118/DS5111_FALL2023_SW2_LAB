""" Build a perceptron for binary classification """

class Perceptron:
    """ Perceptron  """

    def __init__(self, inputs):
        """ Initialize perceptron and create weight matrix """
        self.dummied_inputs = [val + [-1] for val in inputs]
        self._weights = [0.2] * len(self.dummied_inputs[0])

    def train(self, labels):
        """ Train the model with samples of known identity  """
        for _ in range(5000):
            for value, label in zip(self.dummied_inputs, labels):
                label_delta = label - self.predict(value)
                self._weights[value] += .1 * value * label_delta

    def predict(self, value):
        """ Classify new samples """
        if len(value) == 0:
            return None
        value = value + [-1]
        return int(0 < sum([row[0]*row[1] for row in zip(self._weights, value)])) #pylint: disable=consider-using-generator
