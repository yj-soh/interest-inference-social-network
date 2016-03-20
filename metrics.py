import numpy as np

"""
S@K: the average S@K over all testing users can measures the probability of finding a correct interest among the top K recommended, i.e., predicted ones. To be more specific, for each testing user, S@K is assigned to be 1 if a correct interest is ranked in the top K positions of the recommended interests, and 0 otherwise.
"""
def compute_s_k(truth_labels, test_labels, k):
    acc_sk = 0.0
    for i, truth_label in enumerate(truth_labels):
        test_label = test_labels[i]
        sorted_indexes = np.argsort(-test_label)
        for i in range(0, k):
            index = sorted_indexes[i]
            if truth_label[index] == test_label[index] and truth_label[index] == 1:
                acc_sk = acc_sk + 1.0
                break

    return acc_sk / len(test_labels)

def compute_p_k(truth_labels, test_labels, k):
    acc_pk = 0.0
    for i, truth_label in enumerate(truth_labels):
        test_label = test_labels[i]
        sorted_indexes = np.argsort(-test_label)
        num_retrieved = 0.0
        num_relevant = 0.0
        for i in range(0, k):
            index = sorted_indexes[i]
            if test_label[index] == 1:
                num_retrieved = num_retrieved + 1.0
            if truth_label[index] == test_label[index] and truth_label[index] == 1:
                num_relevant = num_relevant + 1.0

        if num_retrieved == 0:
            continue
        acc_pk = acc_pk + (num_relevant / num_retrieved)

    return acc_pk / len(test_labels)

if __name__ == '__main__':
    truth_labels = np.array([[1,1,1,1,1,0], [1,0,1,0,1,0]])
    test_labels = np.array([[0,1,1,1,0,0], [0,1,0,1,1,0]])
    print compute_s_k(truth_labels, test_labels, 2)
    print compute_p_k(truth_labels, test_labels, 4)