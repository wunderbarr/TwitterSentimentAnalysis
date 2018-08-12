import re
import sys
from utils import write_status
from nltk.stem.porter import PorterStemmer

def preprocess_word(word):
    word=word.strip('\'"?!,.():;')#remove punctuation
    word=re.sub(r'(.)\1+', r'\1\1', word)
    # two letter repititions to two letter: sunnnny->sunny
    # consider:!!! suuunny->suunny
    # .:matches any character except a newline
    # ():Defines a marked subexpression
    word=re.sub(r'(-|\')', '', word)
    # need recursively delete??????
    return word

def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-zA-z0-9\._]*$', word) is not None)
    #!!! return false, means valid
    #......we can convert to lowercase before detection

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet=re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet=re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet=re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet=re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ',tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet=re.sub(r'(:-\(|:\s?\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet=re.sub(r'(:,\(|:\'\(|:\"\()', ' EMO_NEG ', tweet)
    return tweet

def preprocess_tweet(tweet):
    processed_tweet=[]
    tweet=tweet.lower() #to lowercase
    tweet=re.sub(r'((www\.[\S]+)|(https?://[\S]+))',' URL ', tweet)
    #match and subtitute URL
    tweet=re.sub(r'@[\S]+', ' USER_MENTION ', tweet)
    #replace @handle
    tweet=re.sub(r'#(\S+)', r' \1 ', tweet)
    #replace hashtag, \1????
    tweet=re.sub(r'\brt\b', '', tweet)
    #remove retweet
    tweet=re.sub(r'\.{2,}', ' ', tweet)
    #replace multiple dots
    tweet=tweet.strip(' "\'')
    #strip space, ", '
    tweet=handle_emojis(tweet)
    tweet=re.sub(r'\s+', ' ', tweet)
    #replace multiple spaces
    words=tweet.split()

    for word in words:
        word=preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word=str(porter_stemmer.stem(word))
            processed_tweet.append(word)
    
    return ' '.join(processed_tweet)#!!!!!! ' '

    
def preprocess_csv(csv_name, processed_name, test_file):
    save_to_file=open(processed_name,'w')

    with open(csv_name, 'r', encoding = "ISO-8859-1") as csv:
        lines=csv.readlines()
        total=len(lines)
        for i, line in enumerate(lines):
            tweet_id=line[:line.find(',')]
            if not test_file:
                line=line[1+line.find(','):]
                label=int(line[:line.find(',')])
            line=line[1+line.find(','):]
            tweet=line
            processed_tweet=preprocess_tweet(tweet)
            if not test_file:
                save_to_file.write('%s, %d, %s\n' % (tweet_id, label, processed_tweet))
            else:
                save_to_file.write('%s, %s\n' % (tweet_id, processed_tweet))
            write_status(i+1, total)
    save_to_file.close()
    print ('\n saved processed tweets to: %s' % processed_name)
    return processed_name

if __name__=='__main__':
    if len(sys.argv)!=2:
        print ('Usage: Python preprocess.py <raw-CSV>')
        exit()
    use_stemmer=False
    csv_name=sys.argv[1]
    processed_name=sys.argv[1][:-4]+'-processed.csv'
    if use_stemmer:
        porter_stemmer=PorterStemmer()
        processed_name=sys.argv[1][:-4]+'-processed-stemmed.csv'
    preprocess_csv(csv_name, processed_name, test_file=True)