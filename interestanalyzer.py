import pickle
import os.path
import numpy as np
import csv
import tweetparser
import linkedinparser
import fbparser
from classifier import Classifier
from featurebuilder import FeatureBuilder

TRAINING_DIRECTORIES = {'fb': 'data/train/Facebook/', 'tweets': 'data/train/Twitter/', 'linkedin': 'data/train/LinkedIn/'}
TESTING_DIRECTORIES = {'fb': 'data/test/Facebook/', 'tweets': 'data/test/Twitter/', 'linkedin': 'data/test/LinkedIn/'}
TRAINING_FILES = {'fb': 'data/generated/training_fb.txt', 'tweets': 'data/generated/training_tweets.txt', 'linkedin': 'data/generated/training_linkedin.txt'}
TESTING_FILES = {'fb': 'data/generated/testing_fb.txt', 'tweets':'data/generated/testing_tweets.txt', 'linkedin': 'data/generated/testing_linkedin.txt'}
TRAINING_FEATURE_FILES = {'fb': 'data/generated/features/training_fb_features.csv', 'tweets': 'data/generated/features/training_tweets_features.csv', 'linkedin': 'data/generated/features/training_linkedin_features.csv'}
TESTING_FEATURE_FILES = {'fb': 'data/generated/features/testing_fb_features.csv', 'tweets': 'data/generated/features/testing_tweets_features.csv', 'linkedin': 'data/generated/features/testing_linkedin_features.csv'}
TRAINING_LABELS_FILE = 'data/train/GroundTruth/groundTruth.txt'
TESTING_LABELS_FILE = 'data/test/GroundTruth/groundTruth.txt'

CLASSIFIER_FILE = 'data/generated/classifer.txt'

class InterestAnalyzer:

    def __init__(self):
        self.feature_builder = FeatureBuilder()

    def rebuild_features(self):
        print 'Parsing Facebook...'
        fbparser.parse({'in': TRAINING_DIRECTORIES['fb'], 'out': TRAINING_FILES['fb']})
        print 'Parsing Twitter...'
        tweetparser.parse({'in': TRAINING_DIRECTORIES['tweets'], 'out': TRAINING_FILES['tweets']})
        print 'Parsing LinkedIn...'
        linkedinparser.parse({'in': TRAINING_DIRECTORIES['linkedin'], 'out': TRAINING_FILES['linkedin']})
        
        print 'Building features...'
        # build features for training data
        self.feature_builder.create_feature_vectors(TRAINING_FILES['linkedin'], TRAINING_FEATURE_FILES['linkedin'], 'linkedin')
        self.feature_builder.create_feature_vectors(TRAINING_FILES['tweets'], TRAINING_FEATURE_FILES['tweets'], 'tweets')
        self.feature_builder.create_feature_vectors(TRAINING_FILES['fb'], TRAINING_FEATURE_FILES['fb'], 'fb')

    def retrain_classifier(self):
        print 'Training classifier...'
        self.classifier = Classifier()

    def save_classifier(self):
        pickle.dump(self.classifier, open(CLASSIFIER_FILE, 'wb'))

    def load_classifier(self):
        self.classifier = pickle.load(open(CLASSIFIER_FILE, 'rb'))

    def classifier_predict(self):
        print 'Parsing Facebook...'
        fbparser.parse({'in': TESTING_DIRECTORIES['fb'], 'out': TESTING_FILES['fb']})
        print 'Parsing Twitter...'
        tweetparser.parse({'in': TESTING_DIRECTORIES['tweets'], 'out': TESTING_FILES['tweets']})
        print 'Parsing LinkedIn...'
        linkedinparser.parse({'in': TESTING_DIRECTORIES['linkedin'], 'out': TESTING_FILES['linkedin']})
        
        print 'Building features...'
        self.feature_builder.create_feature_vectors(TESTING_FILES['linkedin'], TESTING_FEATURE_FILES['linkedin'], 'linkedin')
        self.feature_builder.create_feature_vectors(TESTING_FILES['tweets'], TESTING_FEATURE_FILES['tweets'], 'tweets')
        self.feature_builder.create_feature_vectors(TESTING_FILES['fb'], TESTING_FEATURE_FILES['fb'], 'fb')

        linkedin_testing_features = np.loadtxt(TESTING_FEATURE_FILES['linkedin'], delimiter=',')
        tweets_testing_features = np.loadtxt(TESTING_FEATURE_FILES['tweets'], delimiter=',')
        fb_testing_features = np.loadtxt(TESTING_FEATURE_FILES['fb'], delimiter=',')
        all_testing_features = np.concatenate((fb_testing_features, tweets_testing_features, linkedin_testing_features), axis=1)
        
        print 'Predicting labels...'
        print 'LinkedIn classifier:'
        print self.classifier.predict_testing_data('linkedin', linkedin_testing_features, TESTING_LABELS_FILE, 'results_l.txt')
        print 'Twitter classifier:'
        print self.classifier.predict_testing_data('tweets', tweets_testing_features, TESTING_LABELS_FILE, 'results_t.txt')
        print 'Facebook classifier:'
        print self.classifier.predict_testing_data('fb', fb_testing_features, TESTING_LABELS_FILE, 'results_fb.txt')
        print 'Early fusion classifier:'
        print self.classifier.predict_testing_data('all', all_testing_features, TESTING_LABELS_FILE, 'results.txt')


if __name__ == '__main__':
    print 'Loading required...'
    interestAnalyzer = InterestAnalyzer()

    option = 0
    while option != 4:
        option = input('What do you want to do?\n1. Rebuild Features\n2. Retrain Classifier\n3. Classify Test Tweets\n4. Goodbye\nAnswer: ')

        if option == 1:
            print 'Please wait...'
            interestAnalyzer.rebuild_features()
            print 'Features built!'
        elif option == 2:
            print 'Please wait...'
            interestAnalyzer.retrain_classifier()
            print 'Classifier trained!'
        elif option == 3:
            print 'Please wait...'
            interestAnalyzer.classifier_predict()
