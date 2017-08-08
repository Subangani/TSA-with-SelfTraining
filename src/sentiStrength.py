import nltk

def createDict(fileName, LexiconDict):
    f0=open(fileName,'r')
    line = f0.readline()
    while line:
        row= line.split("\t")
        line=f0.readline()
        LexiconDict.update({row[0]: row[1]})


senti140unigramFile="../resources/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt"
senti140bigramFile="../resources/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt"
senti140pairsFile="../resources/Sentiment140-Lexicon-v0.1/pairs-pmilexicon.txt"

NRC_unigramFile="../resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt"
NRC_bigramFile="../resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt"
NRC_pairsFile="../resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/pairs-pmilexicon.txt"
NRC_hashtagFile="../resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/sentimenthashtags.txt"

sentiWordNetfilename="../resources/SentiWordNet_3.0.0_20130122.txt"
bingLiufilename="../resources/BingLiu.csv"

senti140unigram={}
senti140bigram={}
senti140pairs={}
NRC_unigram={}
NRC_bigram={}
NRC_pairs={}
NRC_hashtag={}
sentiWordnetDict={}
binLiuDict={}


createDict(senti140unigramFile,senti140unigram)
createDict(senti140bigramFile,senti140bigram)
createDict(senti140pairsFile,senti140pairs)
createDict(NRC_unigramFile,NRC_unigram)
createDict(NRC_bigramFile,NRC_bigram)
createDict(NRC_hashtagFile,NRC_hashtag)
createDict(NRC_pairsFile,NRC_pairs)
createDict(bingLiufilename,binLiuDict)

def get_lexiconScore(tweet,lib):
    global senti140pairs,senti140unigram,senti140unigram
    words=tweet.split()
    unigramList=getngramWord(words,1)
    bigramList=getngramWord(words,2)
    uniScore=0.0
    biScore=0.0
    if (lib==1):

        for word in unigramList:
            if  senti140unigram.has_key(word):
                uniScore+=float(senti140unigram.get(word))
        for word in bigramList:
            if senti140bigram.has_key(word):
                biScore+=float(senti140bigram.get(word))
        return uniScore, biScore
    if (lib==2):
        hashScore = 0.0
        for word in unigramList:
            if  NRC_unigram.has_key(word):
                uniScore+=float(NRC_unigram.get(word))
            if (list(word)[0]=='#'):
                ar=list(word)
                ar.remove("#")
                word=''.join(ar)
                if not NRC_hashtag.get(word)==None:
                    if (NRC_hashtag.get(word) =='positive\n'):
                        hashScore+=1
                    elif (NRC_hashtag.get(word)=='negative\n'):
                        hashScore-=1.0
        for word in bigramList:
            if NRC_bigram.has_key(word):
                biScore+=float(NRC_bigram.get(word))
        return uniScore,biScore,hashScore

def getngramWord(words,gram):
    nramList=[]
    for i in range(len(words) + 1 - gram):
        temp = ""
        if not words[i:i + gram] is "":
            if (gram == 1):
                temp = words[i]
            elif (gram == 2):
                temp = words[i] + " " + words[i + 1]
        nramList.append(temp)
    return nramList

def loadSentiWordNet(filename):
    global sentiWordnetDict
    tempDictionary={}
    f0=open(filename,'r')
    line = f0.readline()
    lineNumber=0
    while (line):
        lineNumber+=1

        # If it's a comment, skip this line.
        if not ((line.strip()).startswith("#")):
        # We use tab separation
            data = line.split("\t")
            wordTypeMarker = data[0]
            if (len(data) == 6):
                #Calculate synset score as score = PosS - NegS
                synsetScore = float(data[2])- float(data[3])
                synTermsSplit = data[4].split(" ")


                for synTermSplit in synTermsSplit:
                    synTermAndRank = synTermSplit.split("#")
                    synTerm = synTermAndRank[0] + "#" + wordTypeMarker
                    synTermRank = int(synTermAndRank[1]);

                    if not (tempDictionary.has_key(synTerm)):
                        tempDictionary[str(synTerm)]={}
                    tempDictionary[str(synTerm)][str(synTermRank)]=synsetScore
        line = f0.readline()

    for k1, v1 in tempDictionary.iteritems():
        # Score = 1 / 2 * first + 1 / 3 * second + 1 / 4 * third.....etc.
        # Sum = 1 / 1 + 1 / 2 + 1 / 3...
        score = 0.0
        sum = 0.0
        for k2, v2 in v1.iteritems():
            score += v2/ float(k2)
            sum += 1.0 /float(k2)

        score /= sum

        sentiWordnetDict[k1]=score
    return sentiWordnetDict
loadSentiWordNet(sentiWordNetfilename)


def getSentiWordScore(tweet):
    global sentiWordnetDict
    try:
        nlpos = {}
        nlpos['a'] = ['JJ', 'JJR', 'JJS']  # adjective tags in nltk
        nlpos['n'] = ['NN', 'NNS', 'NNP', 'NNPS']  # nouns ...
        nlpos['v'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN']  # verbs
        nlpos['r'] = ['RB', 'RBR', 'RBS']  # adverbs

        text = tweet.split()
        tags = nltk.pos_tag(text)
        taggedTwits=[]
        for i in range(0,len(tags)):
            if (tags[i][1] in nlpos['a']):
                taggedTwits.append(tags[i][0]+"#a")
            elif (tags[i][1] in nlpos['n']):
                taggedTwits.append(tags[i][0]+"#n")
            elif (tags[i][1] in nlpos['v']):
                taggedTwits.append(tags[i][0]+"#v")
            elif (tags[i][1] in nlpos['r']):
                taggedTwits.append(tags[i][0]+"#r")
        score=0.0
        for i in range(0,len(taggedTwits)):
            if taggedTwits[i] in sentiWordnetDict:
                score+=float(sentiWordnetDict.get(taggedTwits[i]))
        return score
    except:
        None

def getBingLuiScore(tweet):
    global binLiuDict
    score=0.0
    for word in tweet.split():
        if binLiuDict.has_key(word):
            if (binLiuDict.get(word)=='positive\n'):
                score += 1
            if (binLiuDict.get(word)=='negative\n'):
                score -= 1
    return score



