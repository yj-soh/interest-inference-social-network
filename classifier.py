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
        # self.train_classifier('fb')
        # self.train_classifier_all()

    # trains 2 classifiers, one for unigram features, another for lda features
    def train_classifier(self, social_media):
        features = np.loadtxt(TRAINING_FEATURE_FILES[social_media], delimiter=',')
        lda_features = features[:,:20]

        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')

        self.classifiers[social_media + 'unigrams'] = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
        self.classifiers[social_media + 'lda'] = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
        # # self.classifiers[social_media] = RandomForestClassifier(n_estimators=60)
        # self.classifiers[social_media] = KNeighborsClassifier(n_neighbors=25)

        self.classifiers[social_media + 'unigrams'].fit(features, labels)
        self.classifiers[social_media + 'lda'].fit(lda_features, labels)

    # early fusion concatenate all features
    def train_classifier_all(self):
        fb_features = np.loadtxt(TRAINING_FEATURE_FILES['fb'], delimiter=',')
        tweets_features = np.loadtxt(TRAINING_FEATURE_FILES['tweets'], delimiter=',')
        linkedin_features = np.loadtxt(TRAINING_FEATURE_FILES['linkedin'], delimiter=',')

        features = np.concatenate((fb_features, tweets_features, linkedin_features), axis=1)
        labels = np.loadtxt(TRAINING_LABELS_FILE, delimiter=',')

        self.classifiers['all'] = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
        self.classifiers['all'].fit(features, labels)

    def predict_late_fusion_testing_data(self, result_labels_array, truth_file, results_file):
        fused_result_labels = np.zeros(result_labels_array[0].shape)
        social_media_weightage = [0.1] * len(result_labels_array)

        for i, result_labels in enumerate(result_labels_array):
            for j, label in enumerate(result_labels):
                fused_result_labels[j] = fused_result_labels[j] + (label * social_media_weightage[i])

        fused_result_labels[fused_result_labels >= 0.1] = 1
        fused_result_labels[fused_result_labels < 0.1] = 0

        np.savetxt(results_file, fused_result_labels.astype(int), fmt='%i', delimiter=',')
        truth_labels = np.loadtxt(truth_file, delimiter=',')

        precision = metrics.precision_score(truth_labels, fused_result_labels, average='macro')
        recall = metrics.recall_score(truth_labels, fused_result_labels, average='macro')
        f1 = metrics.f1_score(truth_labels, fused_result_labels, average='macro')

        return {'recall': recall, 'precision': precision, 'F1': f1}


    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, social_media, testing_features, truth_file, results_file):
        lda_features = testing_features[:,:20]

        # late fusion of unigrams and lda
        lda_result_labels = self.classifiers[social_media + 'lda'].predict(lda_features)
        unigram_result_labels = self.classifiers[social_media + 'unigrams'].predict(testing_features)

        result_labels_array = [lda_result_labels, unigram_result_labels]

        return self.predict_late_fusion_testing_data(result_labels_array, truth_file, results_file)

if __name__ == '__main__':
    classifier = Classifier()

    linkedin_testing_features = np.loadtxt(TESTING_FEATURE_FILES['linkedin'], delimiter=',')
    tweets_testing_features = np.loadtxt(TESTING_FEATURE_FILES['tweets'], delimiter=',')

    print classifier.predict_testing_data('linkedin', linkedin_testing_features, TESTING_LABELS_FILE, 'results_l.txt')
    print classifier.predict_testing_data('tweets', tweets_testing_features, TESTING_LABELS_FILE, 'results_t.txt')

    tweets_result_labels = np.loadtxt('results_t.txt', delimiter=',')
    linkedin_result_labels = np.loadtxt('results_l.txt', delimiter=',')

    print classifier.predict_late_fusion_testing_data([tweets_result_labels, linkedin_result_labels], TESTING_LABELS_FILE, 'result.txt')
