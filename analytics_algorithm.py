"""This file to define and measure the algorith"""
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from metrics_calculation import MetricsCalculation


metrics_calc = MetricsCalculation()
class AnalyticsAlgorithm(object):

    def decision_tree(self, x_train, y_train, x_test, y_test, dirname):
        """This method to train and predict decision tree algorithm"""
        FILENAME = 'decision_tree_confusion_matrix.png'
        dt_report = {}
        try:
            dt_report['algorithm'] = "DECISION_TREE"
            file_path = os.path.join(dirname, FILENAME)
            decision_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=33)
            decision_classifier.fit(x_train, y_train)
            y_pred = decision_classifier.predict(x_test)
            y_train_pred = decision_classifier.predict(x_train)
            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            # plot_confusion_matrix(decision_classifier, x_test, y_test)
            plt.savefig(file_path)
            dt_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, dt_report)

        except Exception as exec:
            print(exec)
        return dt_report

    def k_nearest_neighbour(self, x_train, y_train, x_test, y_test, dirname):
        """This method to train and predict k_nearest_neighbour algorithm"""
        FILENAME = "k_nearest_neighbour_confusion_matrix.png"
        knn_report = {}
        try:
            knn_report['algorithm'] = "K_NEAREST_NEIGHBOUR"
            file_path = os.path.join(dirname, FILENAME)
            knn_classifier = KNeighborsClassifier(n_neighbors=30)
            knn_classifier.fit(x_train, y_train)
            y_pred = knn_classifier.predict(x_test)
            y_train_pred = knn_classifier.predict(x_train)
            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            # plot_confusion_matrix(knn_classifier, x_test, y_test)
            plt.savefig(file_path)
            knn_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, knn_report)

        except Exception as exec:
            print(exec)

        return knn_report

    def logistic_regression(self, x_train, y_train, x_test, y_test, dirname):
        """This method to train and predict logistic_regression algorithm"""
        FILENAME = "logistic_regression_confusion_matrix.png"
        logistic_reg_report = {}
        try:
            logistic_reg_report['algorithm'] = "LOGISTC_REGRESSION"
            file_path = os.path.join(dirname, FILENAME)
            logistic_classifier = LogisticRegression(max_iter=5)
            logistic_classifier.fit(x_train, y_train)
            y_pred = logistic_classifier.predict(x_test)
            y_train_pred = logistic_classifier.predict(x_train)
            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            # plot_confusion_matrix(logistic_classifier, x_test, y_test)
            plt.savefig(file_path)
            logistic_reg_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, logistic_reg_report)

        except Exception as exec:
            print(exec)

        return logistic_reg_report

    def random_forest(self, x_train, y_train, x_test, y_test, dirname):
        """This method to train and predict random_forest algorithm"""
        FILENAME = "random_forest_confusion_matrix.png"
        rf_report = {}
        try:
            rf_report['algorithm'] = "RANDOM_FOREST"
            file_path = os.path.join(dirname, FILENAME)
            random_forest_classifier = RandomForestClassifier(n_estimators = 1, criterion = 'entropy', random_state = 42)
            random_forest_classifier.fit(x_train, y_train)
            y_pred = random_forest_classifier.predict(x_test)
            y_train_pred = random_forest_classifier.predict(x_train)
            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            # plot_confusion_matrix(random_forest_classifier, x_test, y_test)
            plt.savefig(file_path)
            rf_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, rf_report)

        except Exception as exec:
            print(exec)

        return rf_report

    def support_vector_machine(self, x_train, y_train, x_test, y_test, dirname):
        """This method to train and predict support_vector_machine algorithm"""
        FILENAME = "svm_confusion_matrix.png"
        svm_report = {}
        try:
            svm_report['algorithm'] = "SVM"
            file_path = os.path.join(dirname, FILENAME)
            svc_classifier =SVC(kernel='linear',gamma='auto',C=10, random_state = 0)
            svc_classifier.fit(x_train, y_train)
            y_pred = svc_classifier.predict(x_test)
            y_train_pred = svc_classifier.predict(x_train)
            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            # plot_confusion_matrix(svc_classifier, x_test, y_test)
            plt.savefig(file_path)
            svm_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, svm_report)

        except Exception as exec:
            print(exec)

        return svm_report

    def generate_model(self):
        """This method to generate artificial neural network"""
        model = Sequential()
        model.add(Dense(16, input_dim=23, activation='sigmoid'))
        model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_header(self, header_base, column_count):
        header_list = []
        try:
            for cnt in range(column_count):
                header_list.append(header_base + str(cnt+1))
            return header_list
        except Exception as exec:
            print(exec)

    def artificial_neural_network(self, x_train, y_train, x_test, y_test, dirname):
        """This method used to train artificial neural network"""
        FILENAME = "ann_confusion_matrix.png"
        ann_report = {}
        try:
            ann_report['algorithm'] = "ANN"
            encoder = LabelEncoder()

            encoder.fit(y_train)
            encoder.fit(y_test)
            encoded_train_y = encoder.transform(y_train)
            encoded_test_y = encoder.transform(y_test)
            # convert integers to dummy variables (i.e. one hot encoded)
            y_train = np_utils.to_categorical(encoded_train_y)
            y_test = np_utils.to_categorical(encoded_test_y)
            file_path = os.path.join(dirname, FILENAME)
            model = self.generate_model()

            history = model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=12,
                                validation_data=(x_test, y_test))

            first_layer_weights = model.layers[0].get_weights()[0]
            first_layer_biases = model.layers[0].get_weights()[1]

            column_count = first_layer_weights.shape[1]
            column_values = self.create_header('Layer_1_Node_', column_count)
            layer_1_weights = pd.DataFrame(first_layer_weights, columns=column_values)
            layer_1_weights.to_csv('layer_1_weights.csv')
            layer_1_biases = pd.DataFrame(first_layer_biases, columns=['Layer_1_Biases'])
            layer_1_biases.to_csv('layer_1_biases.csv')

            second_layer_weights = model.layers[1].get_weights()[0]
            second_layer_biases = model.layers[1].get_weights()[1]

            column_count = second_layer_weights.shape[1]
            column_values = self.create_header('Layer_2_Node_', column_count)
            layer_2_weights = pd.DataFrame(second_layer_weights, columns=column_values)
            layer_2_weights.to_csv('layer_2_weights.csv')
            layer_2_biases = pd.DataFrame(second_layer_biases, columns=['Layer_2_Biases'])
            layer_2_biases.to_csv('layer_2_biases.csv')

            final_layer_weights = model.layers[2].get_weights()[0]

            column_count = final_layer_weights.shape[1]
            column_values = self.create_header('final_layer_Node_', column_count)
            final_layer_weights = pd.DataFrame(final_layer_weights, columns=column_values)
            final_layer_weights.to_csv('final_layer_weights.csv')

            # Converting one hot encoding to orginal data
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)

            y_train_pred = model.predict(x_train)
            y_train_pred = np.argmax(y_train_pred, axis=1)

            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)

            metrics_calc.save_confusion_matrix(y_test, y_pred, file_path)
            ann_report = metrics_calc.performance_measure(y_train, y_train_pred, y_test, y_pred, ann_report)
            metrics_calc.training_loss(history)
            metrics_calc.calc_accuracy(history)


        except Exception as exec:
            print(exec)

        return ann_report


