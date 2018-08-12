import utils
TrainProcessedFile='dataset/trainKaggle-processed.csv'
TestProcessedFile='dataset/testKaggle-processed.csv'
PositiveWordsFile='dataset/positive-words.txt'
NegativeWordsFile='dataset/negative-words.txt'
TRAIN=True

def classify (processedCSV, TestFile, **params):
    positive_words=utils.file_to_wordset(params.pop('positive_words'))
    negative_words=utils.file_to_wordset(params.pop('negative_words'))
    predictions=[]
    with open(processedCSV, 'r') as csv:
        for line in csv:
            if TestFile:
                tid, tweet= line.strip().split(',')
            else:
                tid, label, tweet =line.strip().split(',')
            pos_count, neg_count=0,0
            for word in tweet.split():
                if word in positive_words:
                    pos_count+=1
                elif word in negative_words:
                    neg_count+=1
            prediction=1 if pos_count>=neg_count else 0
            if TestFile:
                predictions.append((tid, prediction))
            else:
                predictions.append((tid, int(label), prediction))
    return predictions
if __name__=='__main__':
    if TRAIN:
        predictions=classify(TrainProcessedFile, TestFile=(not TRAIN), positive_words=PositiveWordsFile, negative_words=NegativeWordsFile)
        correct=sum([1 for p in predictions if p[1]==p[2]])*100.0/len(predictions)
        print ('Correct = %.2f%%' % correct)
    else:
        predictions=classify(TestProcessedFile, TestFile=(not TRAIN), positive_words=PositiveWordsFile, negative_words=NegativeWordsFile)
        utils.save_results_to_csv(predictions, 'baseline.csv')