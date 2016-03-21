import numpy as np
import pickle
import kmetrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics

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
        self.train_classifier_all()

    def train_classifier(self, social_media):
        features = np.loadtxt(TRAINING_FEATURE_FILES[social_media], delimiter=',')
        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')

        self.classifiers[social_media] = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
        # # self.classifiers[social_media] = RandomForestClassifier(n_estimators=60)
        # self.classifiers[social_media] = KNeighborsClassifier(n_neighbors=25)
        self.classifiers[social_media].fit(features, labels)

    # early fusion concatenate all features
    def train_classifier_all(self):
        fb_features = np.loadtxt(TRAINING_FEATURE_FILES['fb'], delimiter=',')
        tweets_features = np.loadtxt(TRAINING_FEATURE_FILES['tweets'], delimiter=',')
        linkedin_features = np.loadtxt(TRAINING_FEATURE_FILES['linkedin'], delimiter=',')

        features = np.concatenate((fb_features, tweets_features, linkedin_features), axis=1)
        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')

        self.classifiers['all'] = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
        self.classifiers['all'].fit(features, labels)

    def predict_late_fusion_testing_data(self, fb_result_labels, tweets_result_labels, linkedin_result_labels, truth_file, results_file):
        fused_result_labels = np.zeros(fb_result_labels.shape)
        social_media_weightage = [0.1, 0.1, 0.8]
        social_media_result_labels = [fb_result_labels, tweets_result_labels, linkedin_result_labels]

        for i, result_labels in enumerate(social_media_result_labels):
            for j, label in enumerate(result_labels):
                fused_result_labels[j] = fused_result_labels[j] + (label * social_media_weightage[i])

        np.savetxt(results_file, fused_result_labels, delimiter=',')
        truth_labels = np.loadtxt(truth_file, delimiter=',')

        return kmetrics.compute_p_k(truth_labels, fused_result_labels, 10)


    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, social_media, testing_features, truth_file, results_file):
        result_labels = self.classifiers[social_media].predict(testing_features)
        np.savetxt(results_file, result_labels.astype(int), fmt='%i', delimiter=',')

        truth_labels = np.loadtxt(truth_file, delimiter=',')

        return kmetrics.compute_p_k(truth_labels, result_labels, 10)

if __name__ == '__main__':
    classifier = Classifier()

    linkedin_testing_features = np.loadtxt(TESTING_FEATURE_FILES['linkedin'], delimiter=',')
    tweets_testing_features = np.loadtxt(TESTING_FEATURE_FILES['tweets'], delimiter=',')
    fb_testing_features = np.loadtxt(TESTING_FEATURE_FILES['fb'], delimiter=',')
    all_testing_features = np.concatenate((fb_testing_features, tweets_testing_features, linkedin_testing_features), axis=1)

    print classifier.predict_testing_data('linkedin', linkedin_testing_features, TESTING_LABELS_FILE, 'results_l.txt')
    print classifier.predict_testing_data('tweets', tweets_testing_features, TESTING_LABELS_FILE, 'results_t.txt')
    print classifier.predict_testing_data('fb', fb_testing_features, TESTING_LABELS_FILE, 'results_fb.txt')
    print classifier.predict_testing_data('all', all_testing_features, TESTING_LABELS_FILE, 'results.txt')

    tweets_result_labels = np.loadtxt('results_t.txt', delimiter=',')
    fb_result_labels = np.loadtxt('results_fb.txt', delimiter=',')
    linkedin_result_labels = np.loadtxt('results_l.txt', delimiter=',')

    print classifier.predict_late_fusion_testing_data(fb_result_labels, tweets_result_labels, linkedin_result_labels, TESTING_LABELS_FILE, 'results.txt')