import nltk
import operator
import postag

# extract k most frequent words from the sorted list
def mostFreqList(l,k):
    m=[w[0] for w in l[0:k]]
    return m

#get all the words from sorted list
def getSortedWordCount(processedFile,writeFile,gram):
    d = get_word_features(ngramText(processedFile,gram))
    f1=open(writeFile,"w")
    l = sortList(d)
    f1.write(str(l))
    return l

# from a list of words returns a dictionary with word, freq as key, value
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist[0].split())
    result=[]
    for k in wordlist.keys():
        result.append([k,wordlist[k]])
    return result

# generate vector of unigrams in text file
def ngramText(processedFile,gram):
    textWords=[]
    textWords.append(ngram(processedFile,gram))
    #print textWords
    return textWords

def ngram(processedFile,grams):
    f0 = open(processedFile,"r")
    model =""
    tweetWords = f0.readline()
    postagmodel=""
    while tweetWords:
        tweet = tweetWords.split()
        taggedtweet=postag.posTag(tweet)
        taggedtweetStr=(str(taggedtweet)).replace("[","").replace("]","").replace("'","").replace(" ","").replace("(","").replace(")","").split(",")

        for i in range(0,len(taggedtweetStr)-1,2):
            postagmodel+= taggedtweetStr[i]+":"+taggedtweetStr[i+1]+" "

        length = len(tweet)
        for i in range(length-grams):
            model+=('-'.join(tweet[i:i+grams]))+"  "
        tweetWords=f0.readline()

    if ("positive" in processedFile):
        sentiment = "positive"
    elif ("negative" in processedFile):
        sentiment = "negative"
    elif ("neutral" in processedFile):
        sentiment = "neutral"

    postaggedFile = sentiment + "_" + str(grams)
    postaggedFileName = "../dataset/" + postaggedFile + ".txt"
    f=open(postaggedFileName,"w")
    unigramList = find_unigram_list(postagmodel)
    bigramList = find_bigram_list(postagmodel)
    trigramList = find_trigram_list(postagmodel)

    text=[]
    if (grams==1):
        text.append(str(unigramList).replace("'","").replace(",", " "))
        uni=get_word_features(text)
        f.write(str(sortList(uni)))
    elif (grams==2):
        text.append(str(bigramList).replace("'","").replace(",", " "))
        bi=get_word_features(text)
        f.write(str(sortList(bi)))
    elif (grams==3):
        text.append(str(trigramList).replace("'","").replace(",", " "))
        tri=get_word_features(text)
        f.write(str(sortList(tri)))

    f0.close()
    return model

def sortList(x):
    return list(reversed(sorted(x, key=operator.itemgetter(1))))

def find_unigram_list(tweet):
    return tweet.split()

def find_unigram_taggedList(tweet):
    input_list = tweet.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(", "").replace(")","").split(",")
    unigram_list = []
    for i in range(0,len(input_list) - 1,2):
        unigram_list.append(input_list[i]+":"+input_list[i+1])
    list = str(unigram_list).replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")","").split(",")
    return list

def find_bigram_list(tweet):
  input_list=tweet.split()
  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i]+'-'+input_list[i+1]))
  return bigram_list

def find_bigram_taggedList(tweet):
  input_list=tweet.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")", "").split(",")
  bigram_list = []
  for i in range(0,len(input_list)-3,2):
      bigram_list.append((input_list[i]+':'+input_list[i+1]+"-"+input_list[i+2]+":"+input_list[i+3]))
  list= (str(bigram_list)).replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")", "").split(",")
  return list

def find_trigram_list(tweet):
  input_list= tweet.split()
  trigram_list = []
  for i in range(len(input_list)-2):
      trigram_list.append((input_list[i]+'-'+input_list[i+1]+'-'+input_list[i+2]))
  return trigram_list

def find_trigram_taggedList(tweet):
  input_list= tweet.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")", "").split(",")
  trigram_list = []
  for i in range(0,len(input_list)-5,2):
      trigram_list.append((input_list[i]+':'+input_list[i+1]+'-'+input_list[i+2]+":"+input_list[i+3]+"-"+input_list[i+4]+":"+input_list[i+5]))
  list= str(trigram_list).replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")", "").split(",")
  return list

def getCount(ngram,ngramfile):
    f0 = open(ngramfile, "r")
    ngram_frequency = f0.readline()
    list = ngram_frequency[1:][:-1].replace("[", "").replace("]", "").replace("'", "").replace(" ","").split(',')
    for i in range(0,len(list)-1):
        if (list[i]==ngram):
            return list[i+1]
    return 0

def scoreUnigram(tweet,posuni,neguni,neuuni):
    pos=0
    neg=0
    neu=0
    list=find_unigram_list(tweet)
    for w in list:
        posCount=float(getCount(w,posuni))
        #print posCount
        negCount=float(getCount(w,neguni))
        #print negCount
        neuCount=float(getCount(w,neuuni))
        #print neuCount
        totalCount=posCount+negCount+neuCount
        #print totalCount
        if ( totalCount!= 0):
            pos+=posCount/totalCount
            neg+=negCount/totalCount
            neu+=neuCount/totalCount
    return [pos,neg,neu]
def scoreUnigramPostag(tweet,posuni,neguni,neuuni):
    pos=0
    neg=0
    neu=0
    list=find_unigram_taggedList(tweet)
    for w in list:
        posCount=float(getCount(w,posuni))
        #print posCount
        negCount=float(getCount(w,neguni))
        #print negCount
        neuCount=float(getCount(w,neuuni))
        #print neuCount
        totalCount=posCount+negCount+neuCount
        #print totalCount
        if ( totalCount!= 0):
            pos+=posCount/totalCount
            neg+=negCount/totalCount
            neu+=neuCount/totalCount
    return [pos,neg,neu]

def scoreBigram(tweet,posbi,negbi,neubi):
    pos=0
    neg=0
    neu=0
    list=find_bigram_list(tweet)
    for w in list:
        posCount=float(getCount(w,posbi))
        negCount=float(getCount(w,negbi))
        neuCount=float(getCount(w,neubi))
        totalCount=posCount+negCount+neuCount
        if ( totalCount!= 0):
            pos+=posCount/totalCount
            neg+=negCount/totalCount
            neu+=neuCount/totalCount
    return [pos,neg,neu]

def scoreBigramPostag(tweet,posbi,negbi,neubi):
    pos=0
    neg=0
    neu=0
    list=find_bigram_taggedList(tweet)
    for w in list:
        posCount=float(getCount(w,posbi))
        negCount=float(getCount(w,negbi))
        neuCount=float(getCount(w,neubi))
        totalCount=posCount+negCount+neuCount
        if ( totalCount!= 0):
            pos+=posCount/totalCount
            neg+=negCount/totalCount
            neu+=neuCount/totalCount
    return [pos,neg,neu]

def scoreTrigram(tweet,postri,negtri,neutri):
    pos=0
    neg=0
    neu=0
    list=find_trigram_list(tweet)
    for w in list:
        posCount=float(getCount(w,postri))
        negCount=float(getCount(w,negtri))
        neuCount=float(getCount(w,neutri))
        totalCount=posCount+negCount+neuCount
        if ( totalCount!= 0):
            pos+=posCount/totalCount
            neg+=negCount/totalCount
            neu+=neuCount/totalCount
    return [pos,neg,neu]

def scoreTrigramPostag(tweet, postri, negtri, neutri):
        pos = 0
        neg = 0
        neu = 0
        list = find_trigram_taggedList(tweet)
        for w in list:
            posCount = float(getCount(w, postri))
            negCount = float(getCount(w, negtri))
            neuCount = float(getCount(w, neutri))
            totalCount = posCount + negCount + neuCount
            if (totalCount != 0):
                pos += posCount / totalCount
                neg += negCount / totalCount
                neu += neuCount / totalCount
        return [pos, neg, neu]

# print tweeet
# print find_unigram_list(tweeet)
# print find_bigram_list(tweeet)
# print find_trigram_list(tweeet)
# print scoreUnigram(tweeet,positiveUnigram,negativeUnigram,neutralUnigram)
# print scoreBigram(tweeet,positiveBigram,negativeBigram,neutralBigram)
# print scoreTrigram(tweeet,positiveTrigram,negativeTrigram,neutralTrigram)