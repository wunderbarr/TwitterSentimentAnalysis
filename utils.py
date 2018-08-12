import sys
import pickle
import random
def file_to_wordset(filename):
    words=set()
    with open(filename, 'r', encoding = "ISO-8859-1") as f:
        for line in f:
            words.add(line.strip())
    return words

def write_status(i, total):
    print ('\r Processing %d/%d' % (i, total), flush=True)

def save_results_to_csv(results, csv_file):
    with open(csv_file, 'w') as f:
        f.write('id, prediction\n')
        for tweet_id, pred in results:
            f.write(tweet_id)
            f.write(',')
            f.write(str(pred))
            f.write('\n')

def top_n_words(pkl_file_name, N, shift=0):
    with open(pkl_file_name, 'rb') as pf:
        freq_dist=pickle.load(pf)
    most_common=freq_dist.most_common(N)#[((1, 2), 1), ((3, 4), 1)]
    words={p[0]:i+shift for i, p in enumerate(most_common)} #convert freq to rank
    return words

def top_n_bigrams(pkl_file_name, N, shift=0):#!!!!merge
    with open(pkl_file_name, 'rb') as pf:
        freq_dist=pickle.load(pf)
    most_common=freq_dist.most_common(N)
    bigrams={p[0]:i for i, p in enumerate(most_common)}
    return bigrams

def split_data(tweets, validation_split=0.1):
    index=int((1-validation_split)*len(tweets))
    random.shuffle(tweets)
    return tweets[:index], tweets[index:]