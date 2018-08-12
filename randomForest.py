from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import utils
import random
import numpy as np 


FreqDistFile = "dataset/trainKaggle-processed-freqdist.pkl"
BiFreqDistFile = "dataset/trainKaggle-processed-freqdist-bi.pkl"
TrainProcessedFile = "dataset/trainKaggle-processed.csv"
TestProcessedFile = "dataset/testKaggle-processed.csv"

UnigramSize = 15000
VocabSize = UnigramSize
UseBigrams = False
if UseBigrams:
    BigramSize = 10000
    VocabSize = UnigramSize + BigramSize
FeatType = 'frequency'

def getFeatureVector (tweet):# denoise
    uni_feature_vector=[]
    bi_feature_vector=[]
    words=tweet.split()
    for i in range(len(words)-1):
        word=words[i]
        next_word=words[i+1]
        if unigrams.get(word):
            uni_feature_vector.append(word)
        if UseBigrams:
            if bigrams.get((word, next_word)):
                bi_feature_vector.append((word, next_word))
    if len(words)>=1:
        if unigrams.get(words[-1]):
            uni_feature_vector.append(words[-1])
    return uni_feature_vector, bi_feature_vector

def extractFeatures (tweets, batch_size, test_file, feat_type):
    num_batches = int (np.ceil(len(tweets)/float(batch_size)))#rounding by ceil(), up
    for i in range(num_batches):
        batch=tweets[i*batch_size:(i+1)*batch_size] #no use lilmatrix, sparse
        features=lil_matrix((batch_size, VocabSize))
        labels=np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words=tweet[1][0] #(tweet_id, feature_vector)
                tweet_bigrams=tweet[1][1] #feature vector: uni_feature_vector, bi_feature_vector
            else:
                tweet_words=tweet[2][0] #(tweet_id, int(sentiment), feature_vector)
                tweet_bigrams=tweet[2][1]
                labels[j]=tweet[1]
            if feat_type=='presence':
                tweet_words=set(tweet_words)
                tweet_bigrams=set(tweet_bigrams)
            for word in tweet_words:
                idx = unigrams.get(word) #idx: in dict rank
                if idx:
                    features[j, idx]+=1
            if UseBigrams:
                for bigram in tweet_bigrams:
                    idx=bigrams.get(bigram)
                    if idx:
                        features[j, UnigramSize+idx]+=1
        yield features, labels
def apply_tf_idf(x):
    transformer=TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    transformer.fit(x)
    return transformer

def process_tweets(csv_file, test_file=True):
    tweets=[]
    print ('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines=csv.readlines()
        total=len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet =line.split(',')
            feature_vector=getFeatureVector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            #utils.write_status(i+1,total)
    print ('\n')
    return tweets

if __name__ =='__main__':
    np.random.seed(1337)
    unigrams=utils.top_n_words(FreqDistFile, UnigramSize)
    if UseBigrams:
        bigrams=utils.top_n_bigrams(BiFreqDistFile, BigramSize)
    tweets=process_tweets(TrainProcessedFile, test_file=False)
    #if TRAIN:
    train_tweets, val_tweets = utils.split_data(tweets)#validation
    # else:
    #     random.shuffle(tweets)
    #     train_tweets=tweets
    del tweets
    print ('Extracting Features and Training batches')
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    batch_size=len(train_tweets)#!!!!!!!!!!!!!
    i=1
    n_train_batches=int(np.ceil(len(train_tweets)/float(batch_size)))#!!!!!!!!!!!!!!!
    
    for training_set_x, training_set_y in extractFeatures(train_tweets, test_file=False, feat_type=FeatType, batch_size=batch_size):
        #utils.write_status(i, n_train_batches)
        i+=1
        if FeatType=='frequency':
            tfidf=apply_tf_idf(training_set_x)#.........................
            training_set_x=tfidf.transform(training_set_x)
        clf.fit(training_set_x, training_set_y)#????partial
    print('\n')
    print('Testing')
    del train_tweets

    correct, total =0 , len(val_tweets)
    i=1
    batch_size=len(val_tweets)
    n_val_batches=int(np.ceil(len(val_tweets)/float(batch_size)))
    for val_set_X, val_set_y in extractFeatures(val_tweets, test_file=False, feat_type=FeatType, batch_size=batch_size):
        if FeatType=='frequency':
            val_set_X=tfidf.transform(val_set_X)
        prediction=clf.predict(val_set_X)
        correct+=np.sum(prediction==val_set_y)
        #utils.write_status(i, n_val_batches)
        i+=1
    print ('\nCorrect: %d/%d = %.4f %%' % (correct, total, correct*100./total))

    test_tweets=process_tweets(TestProcessedFile, test_file=True)
    batch_size=len(test_tweets)
    n_test_batches=int(np.ceil(len(test_tweets)/float(batch_size)))
    predictions=np.array([])
    print ('Predicting batches')
    i=1
    for test_set_X, _ in extractFeatures(test_tweets, test_file=True, feat_type=FeatType, batch_size=batch_size):
        if FeatType=='frequency':
            test_set_X=tfidf.transform(test_set_X)
        prediction=clf.predict(test_set_X)
        predictions=np.concatenate((predictions, prediction))
        #utils.write_status(i, n_test_batches)
        i+=1
    predictions=[(str(j), int(predictions[j])) for j in range(len(test_tweets))]
    utils.save_results_to_csv(predictions, 'randomForest.csv')
    print ('\nSaved to randomForest.csv')