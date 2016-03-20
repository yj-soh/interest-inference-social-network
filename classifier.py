import numpy as np
import pickle
import metrics
# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

TRAINING_FEATURE_FILES = {'fb': 'data/generated/features/training_fb_features.csv', 'tweets': 'data/generated/features/training_tweets_features.csv', 'linkedin': 'data/generated/features/training_linkedin_features.csv'}
TESTING_FEATURE_FILES = {'fb': 'data/generated/features/testing_fb_features.csv', 'tweets': 'data/generated/features/testing_tweets_features.csv', 'linkedin': 'data/generated/features/testing_linkedin_features.csv'}
TRAINING_LABELS_FILE = 'data/train/GroundTruth/groundTruth.txt'
TESTING_LABELS_FILE = 'data/test/GroundTruth/groundTruth.txt'

class Classifier:
    def __init__(self):
        self.classifiers = {}
        self.train_classifier('linkedin')
        self.train_classifier('tweets')
        self.train_classifier('fb')

    def train_classifier(self, social_media):
        features = np.loadtxt(TRAINING_FEATURE_FILES[social_media], delimiter=',')
        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')
        # self.classifiers[social_media] = RandomForestClassifier(n_estimators=60)
        self.classifiers[social_media] = KNeighborsClassifier(n_neighbors=40)
        self.classifiers[social_media].fit(features, labels)

    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, social_media, testing_features, truth_file, results_file):
        result_labels = self.classifiers[social_media].predict(testing_features)
        np.savetxt(results_file, result_labels.astype(int), fmt='%i', delimiter=',')

        truth_labels = np.loadtxt(truth_file, delimiter=',')

        return metrics.compute_p_k(truth_labels, result_labels, 10)

if __name__ == '__main__':
    classifier = Classifier()

    linkedin_testing_features = np.loadtxt(TESTING_FEATURE_FILES['linkedin'], delimiter=',')
    tweets_testing_features = np.loadtxt(TESTING_FEATURE_FILES['tweets'], delimiter=',')
    fb_testing_features = np.loadtxt(TESTING_FEATURE_FILES['fb'], delimiter=',')

    print classifier.predict_testing_data('linkedin', linkedin_testing_features, TESTING_LABELS_FILE, 'results_l.txt')
    print classifier.predict_testing_data('tweets', tweets_testing_features, TESTING_LABELS_FILE, 'results_t.txt')
    print classifier.predict_testing_data('fb', fb_testing_features, TESTING_LABELS_FILE, 'results_fb.txt')