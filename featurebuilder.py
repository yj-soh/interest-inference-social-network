import numpy as np
import pickle
import csv

TRAINING_FILES = {'fb': 'data/generated/training_fb.txt', 'tweets': 'data/generated/training_tweets.txt', 'linkedin': 'data/generated/training_linkedin.txt'}
TESTING_FILES = {'fb': 'data/generated/testing_fb.txt', 'tweets':'data/generated/testing_tweets.txt', 'linkedin': 'data/generated/testing_linkedin.txt'}
PHI_FILES = {'fb': 'llda/fb_words.txt_phi', 'tweets': 'llda/tweets_words.txt_phi', 'linkedin': 'llda/linkedin_words.txt_phi'}
INTEREST_WORDS_FILES = {'fb': 'data/generated/fb_interest_words.txt', 'tweets': 'data/generated/tweets_interest_words.txt', 'linkedin': 'data/generated/linkedin_interest_words.txt'}
TRAINING_FEATURE_FILES = {'fb': 'data/generated/features/training_fb_features.csv', 'tweets': 'data/generated/features/training_tweets_features.csv', 'linkedin': 'data/generated/features/training_linkedin_features.csv'}
TESTING_FEATURE_FILES = {'fb': 'data/generated/features/testing_fb_features.csv', 'tweets': 'data/generated/features/testing_tweets_features.csv', 'linkedin': 'data/generated/features/testing_linkedin_features.csv'}
MANUAL_TOPIC_MODEL_FILE = 'resources/manual_topic_model.csv'
WORD2VEC_TOPIC_MODEL_FILE = 'resources/word2vec.txt'

class FeatureBuilder:
    def __init__(self):
        # only use linkedin topic model
        self.interest_words = self.load_llda_interest_words(PHI_FILES['linkedin'], INTEREST_WORDS_FILES['linkedin'])
        self.unigram_feature_dict = {}
        self.build_unigram_feature_dict(TRAINING_FILES['linkedin'], 'linkedin')
        self.build_unigram_feature_dict(TRAINING_FILES['tweets'], 'tweets')
        self.build_unigram_feature_dict(TRAINING_FILES['fb'], 'fb')

    def load_llda_interest_words(self, words_file, output_file):
        # read llda results
        interest_words_phi = []
        for i in range(0, 20):
            interest_words_phi.append([])

        with open(words_file, 'rb') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                interest = row[0]
                word = row[1]
                phi = row[2]
                if interest.isdigit():
                    interest_words_phi[int(interest)].append([word, phi])

        # manual topic model
        with open(MANUAL_TOPIC_MODEL_FILE, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                interest = row[0]
                word = row[2]
                phi = row[3]
                interest_words_phi[int(interest)].append([word, phi])

        # extract top 50 words for each interest
        interest_words = {'lda': [], 'word2vec': []}

        for words_phi in interest_words_phi:
            sorted_words_phi =  sorted(words_phi, key=lambda x: x[1])
            sorted_words_phi.reverse()
            sorted_words_phi = sorted_words_phi[:50]
            words_phi_dict = {}
            for word_phi in sorted_words_phi:
                words_phi_dict[word_phi[0]] = float(word_phi[1])
            interest_words['lda'].append(words_phi_dict)

        # read word2vec results
        interest_words_phi = []
        for i in range(0, 20):
            interest_words_phi.append([])

        with open(words_file, 'rb') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                interest = row[0]
                word = row[1]
                phi = row[2]
                if interest.isdigit():
                    interest_words_phi[int(interest)].append([word, phi])

        # extract top 50 words for each interest
        for words_phi in interest_words_phi:
            sorted_words_phi =  sorted(words_phi, key=lambda x: x[1])
            sorted_words_phi.reverse()
            sorted_words_phi = sorted_words_phi[:50]
            words_phi_dict = {}
            for word_phi in sorted_words_phi:
                words_phi_dict[word_phi[0]] = float(word_phi[1])
            interest_words['word2vec'].append(words_phi_dict)

        return interest_words

    def build_unigram_feature_dict(self, words_file, social_media):
        texts = []
        with open(words_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                texts.append(row)

        unigram_feature_dict = dict()
        unigram_frequencies = dict() # count number of occurrences for particular unigram
        all_unigram_features = []
        count_features = 0
        # process unigram features
        for index, text in enumerate(texts):
            for word in text:
                if (word not in unigram_feature_dict):
                    count_features += 1
                    unigram_feature_dict[word] = count_features - 1
                    unigram_frequencies[word] = 0
                    all_unigram_features.append(word)

        unigram_feature_vectors = np.zeros((len(texts), len(unigram_feature_dict)))

        for index, text in enumerate(texts):
            for word in text:
                if (word in unigram_feature_dict):
                    unigram_frequencies[word] += 1
                    unigram_feature_vectors[index, unigram_feature_dict[word]] += 1

        # select features that occur more than five times
        selected_features = []
        for unigram in all_unigram_features:
            if (unigram_frequencies[unigram] > 5):
                selected_features.append(True)
            else:
                selected_features.append(False)

        # remove unwanted features
        for index, is_feature_selected in reversed(list(enumerate(selected_features))):
            if (not is_feature_selected):
                del all_unigram_features[index]

        # create new dictionary based on selected features
        unigram_feature_dict = dict()
        for index, unigram in enumerate(all_unigram_features):
            unigram_feature_dict[unigram] = index
        
        self.unigram_feature_dict[social_media] = unigram_feature_dict

    def get_unigram_feature_vectors(self, texts, unigram_feature_dict):
        unigram_feature_vectors = np.zeros((len(texts), len(unigram_feature_dict)))
        
        for index, text in enumerate(texts):
            for word in text:
                if (word in unigram_feature_dict):
                    unigram_feature_vectors[index, unigram_feature_dict[word]] = 1 # term presence

        return unigram_feature_vectors

    def create_feature_vectors(self, words_file, features_file='', social_media=''):
        features = []
        texts = []
        with open(words_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                texts.append(row)
                features.append(self.create_interest_feature_vector(row))

        unigram_features = self.get_unigram_feature_vectors(texts, self.unigram_feature_dict[social_media])

        features = np.concatenate((np.array(features), unigram_features), axis=1)

        # save features into csv file
        if features_file != '':
            features = np.asarray(features)
            np.savetxt(features_file, features, delimiter=',')

        return features

    def create_interest_feature_vector(self, words):
        lda_features = [0] * 20
        word2vec_features = [0] * 20

        for interest, words_phi_dict in enumerate(self.interest_words['lda']):
            for word in words:
                if (word in words_phi_dict):
                    lda_features[interest] = lda_features[interest] + words_phi_dict[word]

        for interest, words_phi_dict in enumerate(self.interest_words['word2vec']):
            for word in words:
                if (word in words_phi_dict):
                    word2vec_features[interest] = word2vec_features[interest] + words_phi_dict[word]
                    
        return np.concatenate((np.array(lda_features), np.array(word2vec_features)), axis=0)

if __name__ == '__main__':
    fb = FeatureBuilder()
    fb.create_feature_vectors(TRAINING_FILES['linkedin'], TRAINING_FEATURE_FILES['linkedin'], 'linkedin')
    fb.create_feature_vectors(TESTING_FILES['linkedin'], TESTING_FEATURE_FILES['linkedin'], 'linkedin')
    fb.create_feature_vectors(TRAINING_FILES['tweets'], TRAINING_FEATURE_FILES['tweets'], 'tweets')
    fb.create_feature_vectors(TESTING_FILES['tweets'], TESTING_FEATURE_FILES['tweets'], 'tweets')
    fb.create_feature_vectors(TRAINING_FILES['fb'], TRAINING_FEATURE_FILES['fb'], 'fb')
    fb.create_feature_vectors(TESTING_FILES['fb'], TESTING_FEATURE_FILES['fb'], 'fb')
