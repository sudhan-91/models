"""This file is used to measure the performance of the algorithm"""
import math
from numpy import mean
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
class MetricsCalculation(object):
    """This class used to calculate metrics"""

    def training_loss(self, history):
        """This method used measure the loss of trainig"""
        try:
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'y', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig("training.png")
            plt.show()

        except Exception as exep:
            print("Error occurs in training_loss", exep)

    def calc_accuracy(self, history):
        """This method used measure the loss of trainig"""
        try:
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig("accuracy.png")
            plt.show()
        except Exception as exep:
            print("Error occurs in training_loss", exep)

    def save_confusion_matrix(self, y_test, y_pred, filepath):
        """This method is used save the confusion matrix of neural network"""
        try:
            cnf_matrix = confusion_matrix(y_test, y_pred)
            #Dataset 1
            # classes = ['cluster', 'migraine', 'migraine']
            # Dataset 2
            classes = ["Basilar-type aura", "Familial hemiplegic migraine", "Migraine without aura", "Others", "Sporadic hemiplegic migraine", "Typical aura with migraine", "Typical aura without migraine"]
            # Dataset 3
            # classes = ["Headache", "No_Headache"]
            # Dataset 4
            # classes = ["Migraine", "No_Migraine"]
            # Dataset 5
            # classes = ["No_Migraine", "Migraine"]
            df_cfm = pd.DataFrame(cnf_matrix, index=classes, columns=classes)
            plt.figure(figsize=(10, 7))
            plt.title("Confusion With Matrix Filtered Feature")
            cfm_plot = sn.heatmap(df_cfm, annot=True, cmap='RdBu_r')
            cfm_plot.figure.savefig(filepath)
        except Exception as exep:
            print("Error occurs in save_confusion_matrix", str(exep))


    def calculate_precision(self, y_test, y_pred):
        """This method is used calculate precision"""
        try:
            cnf_matrix = confusion_matrix(y_test, y_pred)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

        except Exception as exep:
            print("Error occurs in calculate_precision", str(exep))

        return FP, FN, TP, TN

    def performance_measure(self, y_train, y_train_pred, y_test, y_pred, report):
        """This method to measure the accuracy"""
        try:
            # y_test = y_test.to_numpy(dtype='int32')
            # y_train = y_train.to_numpy(dtype='int32')
            FP, FN, TP, TN = self.calculate_precision(y_test, y_pred)
            report['trainAccuracy'] = accuracy_score(y_train, y_train_pred) * 100
            report['testAccuracy'] = accuracy_score(y_test, y_pred) * 100
            report['precision'] = mean(TP / (TP + FP)) * 100
            report['sensitivity'] = mean(TP / (TP + FN)) * 100
            report['f-measure'] = mean(2 * ((TP / (TP + FP)) * (TP / (TP + FN))) / \
                                           ((TP / (TP + FP)) + (TP / (TP + FN)))) * 100
            report['specificity'] = mean(TN / (TN + FP)) * 100

        except Exception as exec:
            print(exec)

        return report

    def calculate_time_complexity(self, hyper_params, no_of_samples, dimension):
        """This method used to calculate time complexity if the algorithm"""
        time_complexity = {}
        try:
            # Decision tree formula O(n*log(n)*d)
            time_complexity['decisionTree'] = no_of_samples * (math.log(no_of_samples)) * dimension

            # KNN formula O(knd)
            time_complexity['kNearestNeighbour'] = hyper_params['kNearestNeighbour'][0]['nNeighbors'] \
                                                   * no_of_samples * dimension

            # Logistic regression formula O(nd)
            time_complexity['logisticRegression'] = no_of_samples * dimension

            # Random Forest O(n*log(n)*d*k)
            time_complexity['randomForest'] = no_of_samples * (math.log(no_of_samples)) * dimension * \
                                         hyper_params['randomForest'][0]['nEstimators']

            # SVM formula O(n²)
            time_complexity['svm'] = no_of_samples * no_of_samples

            # Neural network formula (𝑛𝑡∗(𝑖𝑗+𝑗𝑘)) for three layers
            time_complexity['neuralNetwork'] = hyper_params['neuralNetwork'][0]['epochs'] * no_of_samples *\
                        ( hyper_params['neuralNetwork'][0]['firstLayerNode'] + hyper_params['neuralNetwork'][0]['secondLayerNode'] + \
                          hyper_params['neuralNetwork'][0]['secondLayerNode'] + hyper_params['neuralNetwork'][0]['finalLayerNode'])

        except Exception as exep:
            print("Error Occurs in calculate_time_complexity", str(exep))

        return time_complexity
