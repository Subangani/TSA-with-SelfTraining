import re
import csv
import string

import time

#start getStopWordList
def loadStopWordList():
    fp =  open("../resources/stopWords.txt",'r')
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def removeStopWords(tweet,stopWords):
    result=''
    for w in tweet:
        if w[-4:]== "_NEG" :
                if w[:-4] in stopWords:
                         None
                else:
                         result=result+w+' '
        else:
                if w in stopWords:
                        None
                else:
                    result = result + w + ' '
    return result

def negate(tweets):
    fn =open("../resources/negation.txt","r")
    line = fn.readline()
    negationList=[]
    while line:
        negationList.append(line.split(None, 1)[0]);
        line = fn.readline()
    fn.close()
    puncuationMarks = [".", ":", ";", "!", "?"]

    for i in range(len(tweets)):
        if tweets[i] in negationList:
            j = i + 1
            while j < len(tweets):
                if (tweets[j][-1] not in puncuationMarks):
                    tweets[j] = tweets[j] + "_NEG"
                    j = j + 1
                elif(tweets[j][-1] not in puncuationMarks):
                    tweets[j] = tweets[j][-1] + "_NEG"
                else:
                    break
            i = j
    return tweets

#start loading slangs list from file
def loadInternetSlangsList():
    fi=open('../resources/internetSlangs.txt','r')
    slangs={}

    line=fi.readline()
    while line:
        l=line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]]=l[1][:-2]
        line=fi.readline()
    fi.close()
    return slangs

#start replace slangs
def replaceSlangs(tweet,slangsList):
    result=''
    words=tweet.split()
    for w in words:
        if w in slangsList.keys():
            result=result+slangsList[w]+" "
        else:
            result=result+w+" "
    return result

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def preProcessTweet(tweet): # arg tweet, stopWords list and internet slangs dictionnary
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))','url',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','at_user',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # removing sepcial caracter
    tweet = tweet.strip('\'"')
    # replace multi-occurences by two
    processedTweet = replaceTwoOrMore(tweet)
    #replace slangs
    slangs = loadInternetSlangsList()
    words=replaceSlangs(processedTweet,slangs).split()
    #negate tweets
    negatedTweets=negate(words)
    #remove stop words
    stopWords = loadStopWordList()
    preprocessedtweet = removeStopWords(negatedTweets,stopWords)
    # remove punctuations
    puctuationRemovedTweets = ' '.join(word.strip(string.punctuation) for word in preprocessedtweet.split())
    return puctuationRemovedTweets


def process(filename, processedFile):
    f0=open(filename,"r")
    f1 = open(processedFile,"w")
    reader = csv.reader(f0)
    for row in reader:
        a = row[2]
        tweet= preProcessTweet(a)
        f1.write(str(tweet)+"\n")
    f0.close()
    f1.close()

#pre-processing positive twits
positiveFilename= "../dataset/full_data/posTrain.csv"
positiveProcessedfile="../dataset/positiveProcessed.txt"
print "preprocessing positive tweets"
#process(positiveFilename,positiveProcessedfile)

#pre-processing positive twits
negativeFilename= "../dataset/full_data/negTrain.csv"
negativeProcessedfile="../dataset/negativeProcessed.txt"
print "preprocessing negative tweets"
#process(negativeFilename,negativeProcessedfile)

#pre-processing positive twits
neutralFilename= "../dataset/full_data/neuTrain.csv"
neutralProcessedfile="../dataset/neutralProcessed.txt"
print "preprocessing neutral tweets"
#process(neutralFilename,neutralProcessedfile)

