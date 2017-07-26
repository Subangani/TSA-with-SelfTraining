
from docutils.parsers import null
import lexicon
import ngramGenerator
import writingStyle
import preprocess
#import microbloggingFeature

import csv
import warnings
warnings.filterwarnings('ignore')

posLexicon="../resources/positive.txt"
negLexicon="../resources/negative.txt"
#emoticonDict=microbloggingFeature.createEmoticonDictionary("../resources/emoticon.txt")

positiveProcessedfile="../dataset/positiveProcessed.txt"
negativeProcessedfile="../dataset/negativeProcessed.txt"
neutralProcessedfile="../dataset/neutralProcessed.txt"

positiveFile="../dataset/posTrain.csv"
negativeFile="../dataset/negTrain.csv"
neutralFile="../dataset/neuTrain.csv"

positiveUnigramList=null
negativeUnigramList=null
neutralUnigramList=null
positiveBigramList=null
negativeBigramList=null
neutralBigramList=null
positiveTrigramList=null
negativeTrigramList=null
neutralTrigramList=null

def ngramGeneratora():
    global positiveUnigramList,negativeUnigramList,neutralUnigramList,positiveBigramList,neutralTrigramList
    global negativeBigramList,neutralBigramList,positiveTrigramList,positiveTrigramList,negativeTrigramList
    print 'creating ngram lists'
    positiveUnigramList=ngramGenerator.ngram(positiveProcessedfile,1)
    negativeUnigramList=ngramGenerator.ngram(negativeProcessedfile,1)
    neutralUnigramList=ngramGenerator.ngram(neutralProcessedfile,1)
    positiveBigramList=ngramGenerator.ngram(positiveProcessedfile,2)
    negativeBigramList=ngramGenerator.ngram(negativeProcessedfile,2)
    neutralBigramList=ngramGenerator.ngram(neutralProcessedfile,2)
    positiveTrigramList=ngramGenerator.ngram(positiveProcessedfile,3)
    negativeTrigramList=ngramGenerator.ngram(negativeProcessedfile,3)
    neutralTrigramList=ngramGenerator.ngram(neutralProcessedfile,3)

ngramGeneratora()

def mapTweet(tweet):
    vector = []
    preprocessed_tweet=preprocess.preProcessTweet(tweet)
    vector.append(lexicon.getLexiconScore(preprocessed_tweet,posLexicon,negLexicon))
    #vector.append(microbloggingFeature.emoticonScore(preprocessed_tweet,emoticonDict))
    vector.append(writingStyle.uppercasedWordsInTweet(tweet))
    vector.append(writingStyle.exclamationCount(tweet))
    vector.append(writingStyle.questionMarkCount(tweet))
    vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],1))
    vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],2))
    vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],3))
    vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[1],negativeUnigramList[1],neutralUnigramList[1],1))
    return vector

def loadMatrix(posfilename, neufilename, negfilename, poslabel, neulabel, neglabel):
    vectors = []
    labels = []
    kpos = 0
    kneg = 0
    kneu = 0
    print "Loading training dataset..."
    f = open(posfilename, 'r')
    reader = csv.reader(f)
    for row in reader:
        try:
            a = row[2]
            kpos += 1
            z = mapTweet(a)
            vectors.append(z)
            labels.append(float(poslabel))
        except:
            None

    print str(kpos)+" positive lines loaded : "
    f.close()

    f = open(neufilename, "r")
    reader = csv.reader(f)
    for row in reader:
        try:
            a = row[2]
            kneu += 1
            z = mapTweet(a)
            vectors.append(z)
            labels.append(float(neulabel))

        except:
            None
    print str(kneu) + " neutral lines loaded : "
    f.close()

    f = open(negfilename, 'r')
    reader = csv.reader(f)
    for row in reader:
        try:
            a = row[2]
            kneg = kneg + 1
            z = mapTweet(a)
            vectors.append(z)
            labels.append(float(neglabel))
        except:
            None

    f.close()
    print str(kneg) + " negative lines loaded : "
    print "Loading done."
    return vectors, labels




