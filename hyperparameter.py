"""This file is used to set the hyper parameters"""

class Hyperparameter(object):
    """This class used to set the hyper parameter for various algorithm"""

    def set_hyperparameter(self):
        """Initialize hyperparameter for training"""
        try:
            hyper_params = {
                    "decisionTree": [{
                        "criterion": "entropy",
                        "maxDepth": 3,
                        "randomState": 33
                    }],
                    "kNearestNeighbour": [{
                        "nNeighbors": 30
                    }],

                    "logisticRegression": [{
                        "maxIter": 5
                    }],
                    "randomForest": [{
                        "nEstimators": 1,
                        "criterion": "entropy",
                        "randomState": 42
                    }],
                    "svm": [{
                        "kernel": "linear",
                        "gamma": "auto",
                        "C": 10,
                        "randomState": 0
                    }],
                    "neuralNetwork": [{
                        "inputDim": 39,
                        "firstLayerNode": 16,
                        "firstLayerActivation": "sigmoid",
                        "secondLayerNode": 8,
                        "secondLayerActivation": "sigmoid",
                        "finalLayerNode": 3,
                        "finalLayerActivation": "softmax",
                        "loss": "categorical_crossentropy",
                        "optimizer": "adam",
                        "epochs":100
                    }]
                            }

        except Exception as exep:
            print("Error occurs in set_hyperparameter", str(exep))

        return hyper_params

