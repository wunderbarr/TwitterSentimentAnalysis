import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
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
UseBigrams = True
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
        features=np.zeros((batch_size, VocabSize))
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

def build_model():
    model = Sequential()
    model.add(Dense(1, input_dim = VocabSize, activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_model(model, val_tweets):
    correct, total = 0, len(val_tweets)
    for val_set_X, val_set_y in extractFeatures(val_tweets, 500, feat_type=FeatType, test_file=False):
        prediction = model.predict_on_batch(val_set_X)
        prediction = np.round(prediction) 
        #numpy rounds to the nearest even value:This is specifically called Round half to even and is useful because it does not introduce bias. 
        # This is especially important in finance, which it's sometimes called "bankers' rounding" 
        correct += np.sum(prediction == val_set_y[:, None])#!!!!!!!!!!!!!!!!!!!
    return float(correct)/total

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
    nb_epochs = 20
    batch_size=500#!!!!!!!!!!!!!
    
    model = build_model()
    n_train_batches=int(np.ceil(len(train_tweets)/float(batch_size)))#!!!!!!!!!!!!!!!
    best_val_acc = 0.0
    for j in range(nb_epochs):
        i=1
        for training_set_x, training_set_y in extractFeatures(train_tweets, test_file=False, feat_type=FeatType, batch_size=batch_size):
            output = model.train_on_batch(training_set_x, training_set_y)#!!!!!~~~~~~~~~~~~~~
            sys.stdout.write('\rIteration %d %d, loss: %.4f, acc: %.4f' % (i, n_train_batches, output[0], output[1]))#!!!!!!!!!!!!!!!~~~~~~~~~
            sys.stdout.flush()
            i+=1
        val_acc =evaluate_model(model, val_tweets)
        print ('\nEpoch: %d, val_acc: %.4f' % (j+1, val_acc))
        random.shuffle(train_tweets)
        if val_acc>best_val_acc:
            print ('Accuracy improved from %.4f to %.4f, saving model' % (best_val_acc, val_acc))
            best_val_acc = val_acc
            model.save('best_model.h5')
    print('\nTesting')
    del train_tweets
    del model
    model = load_model('best_model.h5')

    test_tweets=process_tweets(TestProcessedFile, test_file=True)
    n_test_batches=int(np.ceil(len(test_tweets)/float(batch_size)))
    predictions=np.array([])
    print ('Predicting batches')
    i=1
    for test_set_X, _ in extractFeatures(test_tweets, test_file=True, feat_type=FeatType, batch_size=batch_size):
        prediction=np.round(model.predict_on_batch(test_set_X).flatten())#~~~~~~~~~~~~~~~~~~~~
        predictions=np.concatenate((predictions, prediction))
        #utils.write_status(i, n_test_batches)
        i+=1
    predictions=[(str(j), int(predictions[j])) for j in range(len(test_tweets))]
    utils.save_results_to_csv(predictions, 'logistic.csv')
    print ('\nSaved to logistic.csv')