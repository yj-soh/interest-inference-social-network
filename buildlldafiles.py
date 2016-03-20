import csv

GROUND_TRUTH_FILE = 'data/train/GroundTruth/groundTruth.txt'
TRAINING_LABELS_FILE = 'llda/labels.txt'

TRAINING_INPUT_FILES = ['data/generated/training_fb.txt', 'data/generated/training_tweets.txt', 'data/generated/training_linkedin.txt']
TRAINING_OUTPUT_FILES = ['llda/fb_words.txt', 'llda/tweets_words.txt', 'llda/linkedin_words.txt']

# read labels
interest_labels = []
with open(GROUND_TRUTH_FILE, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        labels = []
        for label_index, label in enumerate(row):
            if label == '1':
                labels.append(label_index)
        interest_labels.append(labels)

labels_file = open(TRAINING_LABELS_FILE, 'w')
for i, row in enumerate(interest_labels):
    labels_file.write(str(i) + '\t' + '\t'.join(str(v) for v in row))
    labels_file.write('\n')
labels_file.close()

# read data
for i, input_file in enumerate(TRAINING_INPUT_FILES):
    data = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            data.append(row)

    output_file = TRAINING_OUTPUT_FILES[i]
    words_file = open(output_file, 'w')
    for i, row in enumerate(data):
        words_file.write(str(i) + '\t' + '\t'.join(data[i]))
        words_file.write('\n')
    words_file.close()