# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        # loop through the iterations
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            # loop through all the trainingdata
            for i in range(len(trainingData)):
                # get the current trainingdata as f
                f = trainingData[i]      
                # instantiate a counter                    
                vectors = util.Counter()
                # foreach legal label
                for l in self.legalLabels:
                    # add the weigts * f to the counter
                    vectors[l] = self.weights[l] * f
                # get the argMax from the created counter
                y1 = vectors.argMax()
                # get the label we want to compare to
                y2 = trainingLabels[i]
                # if the y1 and y2 are equal to eachother, we do not need to change the weights, so continue to the next piece of data
                if y1 == y2:
                    continue
                # else change the weights
                self.weights[y2] += f
                self.weights[y1] -= f

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        # get the weights
        weights = self.weights[label]

        # sort the keys on values
        keys = sorted(weights, key=weights.get)

        # keys are sorted from lowest to highest value, so reverse
        list.reverse(keys)

        # return the first 100 keys, so the 100 keys with the highest values
        return keys[:100]