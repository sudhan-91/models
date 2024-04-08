"""This file used to measure perfomance and accuracy of machine learning algorithm"""
import pandas as pd
import os
import time
import tracemalloc
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from analytics_algorithm import AnalyticsAlgorithm
from hyperparameter import Hyperparameter
from metrics_calculation import MetricsCalculation


algorithm = AnalyticsAlgorithm()
hyperparams = Hyperparameter()
metrics_calc = MetricsCalculation()
class AlgorithmBenchmarkWrapper(object):
    """This class used to take metics of machine learning algorithm"""

    def read_dataset(self, file_path):
        """This method used to read and encode data"""
        try:
            data = pd.read_csv(file_path)
            data = data.apply(LabelEncoder().fit_transform)
            input_feature = data.drop(['CLASS'], axis=1)
            output_label = data['CLASS']
        except Exception as exec:
            print(exec)

        return input_feature, output_label

    def algorithm_benchmark(self, x_train, y_train, x_test, y_test, time_complexity):
        """This method to measure the performance of various algorithm"""
        CONF_DIR = 'confusion_matrix'
        try:
            time_log = str(round(time.time() * 1000))
            dirname = os.path.join(os.path.dirname(os.getcwd()), CONF_DIR)
            os.makedirs(dirname, exist_ok=True)

            # Decision Tree metrics analysis
            tracemalloc.start()
            start_time = time.time()
            dt_report = algorithm.decision_tree(x_train, y_train, x_test, y_test, dirname)
            dt_report['executionTime'] = (time.time() - start_time)
            dt_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            dt_report['timeComplexity'] = time_complexity['decisionTree']
            tracemalloc.stop()

            # Decision KNN metrics analysis
            tracemalloc.start()
            start_time = time.time()
            knn_report = algorithm.k_nearest_neighbour(x_train, y_train, x_test, y_test, dirname)
            knn_report['executionTime'] = (time.time() - start_time)
            knn_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            knn_report['timeComplexity'] = time_complexity['kNearestNeighbour']
            tracemalloc.stop()

            # Logistic regression metrics analysis
            tracemalloc.start()
            start_time = time.time()
            logistic_report = algorithm.logistic_regression(x_train, y_train, x_test, y_test, dirname)
            logistic_report['executionTime'] = (time.time() - start_time)
            logistic_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            logistic_report['timeComplexity'] = time_complexity['logisticRegression']
            tracemalloc.stop()

            # Random forest metrics analysis
            tracemalloc.start()
            start_time = time.time()
            random_forest_report = algorithm.random_forest(x_train, y_train, x_test, y_test, dirname)
            random_forest_report['executionTime'] = (time.time() - start_time)
            random_forest_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            random_forest_report['timeComplexity'] = time_complexity['randomForest']
            tracemalloc.stop()

            # SVM metrics analysis
            tracemalloc.start()
            start_time = time.time()
            svm_report = algorithm.support_vector_machine(x_train, y_train, x_test, y_test, dirname)
            svm_report['executionTime'] = (time.time() - start_time)
            svm_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            svm_report['timeComplexity'] = time_complexity['svm']
            tracemalloc.stop()

            # Artificial Neural Network metrics analysis
            tracemalloc.start()
            start_time = time.time()
            ann_report = algorithm.artificial_neural_network(x_train, y_train, x_test, y_test, dirname)
            ann_report['executionTime'] = (time.time() - start_time)
            ann_report['memoryUsageInBytes'] = (tracemalloc.get_tracemalloc_memory())
            ann_report['timeComplexity'] = time_complexity['neuralNetwork']
            tracemalloc.stop()

            # merge all metrics
            consolidate_metrics = [dt_report, knn_report, logistic_report,\
                                  random_forest_report, svm_report, ann_report]

        except Exception as exec:
            print(exec)

        return consolidate_metrics

    def convert_into_dataframe(self, consolidate_metrics, dataset_info):
        """This method used to convert metrics into dataframe"""
        combined_data = OrderedDict()
        heading_list = ['ALGORITHM', 'TOTAL_SAMPLE', 'TRAINING_SAMPLE', \
                        'TESTING_SAMPLE', 'FEATURES', 'TRAINING_ACCURACY',\
                        'TESTING_ACCURACY', 'PRECISION', 'F-MEASURE', \
                        'SPECIFICITY', 'SENSITIVITY', 'EXECUTION_TIME', \
                        'MEMEORY_USAGE_IN_BYTES', 'TIME_COMPLEXITY']
        try:
            for heading in heading_list:
                combined_data[heading] = []
                for metrics in consolidate_metrics:
                    if heading == 'ALGORITHM':
                        combined_data[heading].append(metrics['algorithm'])
                    elif heading == 'TOTAL_SAMPLE':
                        combined_data[heading].append(dataset_info['totalSamples'])
                    elif heading == 'TRAINING_SAMPLE':
                        combined_data[heading].append(dataset_info['trainSamples'])
                    elif heading == 'TESTING_SAMPLE':
                        combined_data[heading].append(dataset_info['testSamples'])
                    elif heading == 'FEATURES':
                        combined_data[heading].append(dataset_info['noOfFeature'])
                    elif heading == 'TRAINING_ACCURACY':
                        combined_data[heading].append(metrics['trainAccuracy'])
                    elif heading == 'TESTING_ACCURACY':
                        combined_data[heading].append(metrics['testAccuracy'])
                    elif heading == 'PRECISION':
                        combined_data[heading].append(metrics['precision'])
                    elif heading == 'F-MEASURE':
                        combined_data[heading].append(metrics['f-measure'])
                    elif heading == 'SPECIFICITY':
                        combined_data[heading].append(metrics['specificity'])
                    elif heading == 'SENSITIVITY':
                        combined_data[heading].append(metrics['sensitivity'])
                    elif heading == 'EXECUTION_TIME':
                        combined_data[heading].append(metrics['executionTime'])
                    elif heading == 'MEMEORY_USAGE_IN_BYTES':
                        combined_data[heading].append(metrics['memoryUsageInBytes'])
                    elif heading == 'TIME_COMPLEXITY':
                        combined_data[heading].append(metrics['timeComplexity'])

            consolidated_datafrae = pd.DataFrame(combined_data)
            consolidated_datafrae.to_csv("merged.csv", index=False)

        except Exception as exep:
            print("Error Occurs in convert_into_dataframe", exep)

    def invoke_benchmarking(self, input_file):
        """This method used to benchmark data"""
        dataset_info = {}
        try:
            input_feature, output_label = self.read_dataset(input_file)
            x_train, x_test, y_train, y_test = train_test_split(\
                input_feature, output_label, test_size = 0.15, random_state = 42)
            hyper_params = hyperparams.set_hyperparameter()
            time_complexity = metrics_calc.calculate_time_complexity(hyper_params, len(x_train), input_feature.shape[1])
            dataset_info['totalSamples'] = input_feature.shape[0]
            dataset_info['noOfFeature'] = x_train.shape[1]
            dataset_info['trainSamples'] = len(x_train)
            dataset_info['testSamples'] = len(x_test)
            consolidate_metrics = self.algorithm_benchmark(x_train, y_train, x_test,y_test, time_complexity)
            self.convert_into_dataframe(consolidate_metrics, dataset_info)
        except Exception as exec:
            print(exec)

benchmark = AlgorithmBenchmarkWrapper()
print(os.path.dirname(os.getcwd()))
# input_file = os.path.join(os.path.dirname(os.getcwd()), 'dataset', 'migbase_dataset.csv')
dataset= ['CHISQUARE', 'ENTROPY']
input_file = "E:\Projects\Genetic\Final_report\\1_Migbase_Dataset\\2_T-Test\migbase_dataset-t-test_filtered.csv"
benchmark.invoke_benchmarking(input_file)