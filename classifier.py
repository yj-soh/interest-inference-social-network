import numpy as np
import pickle
# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

TRAINING_FEATURE_FILES = {'linkedin': 'data/generated/features/training_linkedin_features.csv'}
TESTING_FEATURE_FILES = {'linkedin': 'data/generated/features/testing_linkedin_features.csv'}
TRAINING_LABELS_FILE = 'data/train/GroundTruth/groundTruth.txt'

class Classifier:
    def __init__(self):
        self.classifiers = {}
        self.train_classifier('linkedin')

    def train_classifier(self, social_media):
        features = np.loadtxt(TRAINING_FEATURE_FILES[social_media], delimiter=',')
        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')
        self.classifiers[social_media] = RandomForestClassifier(n_estimators=20)
        self.classifiers[social_media].fit(features, labels)

    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, social_media, testing_features, results_file):
        result_labels = self.classifiers[social_media].predict(testing_features)
        np.savetxt(results_file, result_labels.astype(int), fmt='%i', delimiter=',')

if __name__ == '__main__':
    classifier = Classifier()

    testing_features = np.loadtxt(TESTING_FEATURE_FILES['linkedin'], delimiter=',')
    classifier.predict_testing_data('linkedin', testing_features, 'results.txt')