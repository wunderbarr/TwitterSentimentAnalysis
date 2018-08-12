import nltk
import random
import pickle
import sys
import numpy as np
import utils 

TrainProcessedFile = "dataset/trainKaggle-processed.csv"
TestProcessedFile = "dataset/testKaggle-processed.csv"
USE_BIGRAMS = False


def get_data_from_file(fileName, isTrain):  # !!!!!!!!!
    data = []
    with open(fileName, 'r') as csv:  
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if isTrain:
                tag = line.split(',')[1]
                bagOfWords = line.split(',')[2].split()
                if USE_BIGRAMS:
                    bagOfWordsBigram = list(
                        nltk.bigrams(line.split(',')[2].split()))
                    bagOfWords = bagOfWords + bagOfWordsBigram
            else:
                tag = '5'
                bagOfWords = line.split(',')[1].split()
                if USE_BIGRAMS:
                    bagOfWordsBigram = list(
                        nltk.bigrams(line.split(',')[1].split()))
                    bagOfWords = bagOfWords + bagOfWordsBigram
            data.append((bagOfWords, tag))
    return data

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])

if __name__ == '__main__':
    np.random.seed(1337)
    trainCSVfile = TrainProcessedFile
    testCSVfile = TestProcessedFile
    train_data = get_data_from_file(trainCSVfile, isTrain=True) #[(bagOfWords, tag).....]
    trainSet, validationSet = utils.split_data(train_data)
    trainSetFormatted = [(list_to_dict(element[0]), element[1]) for element in trainSet]
    validationSetFormatted = [(list_to_dict(element[0]), element[1]) for element in validationSet]
    numIteration = 1
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[1]
    classifier = nltk.MaxentClassifier.train(trainSetFormatted, algorithm, max_iter= numIteration)
    classifier.show_most_informative_features(10)
    count = int (0)
    for review in validationSetFormatted:
        label = review[1]
        text = review[0]
        determined_label = classifier.classify(text)
        if determined_label!=label:
            count+=1
    accuracy = (len(validationSet)-count)/len(validationSet)
    print ('Validation set accuracy: %.4f' % (accuracy))
    f = open('maxEntClassifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    print ('\n Predicting for test data')
    test_data = get_data_from_file(testCSVfile, isTrain=False)
    testSetFormatted = [(list_to_dict(element[0]),element[1]) for element in test_data]
    tweet_id = int(0)
    result = []
    for review in testSetFormatted:
        text = review[0]
        label = classifier.classify(text)
        result.append((str(tweet_id), label))
        tweet_id += int(1)
    utils.save_results_to_csv(result, 'maxent.csv')
    print ('Saved to maxent.csv')

