import sys
from docutils.parsers import null
import lexicon
import ngramGenerator
import writingStyle
import preprocess
import microbloggingFeature
import sentiStrength
import csv
import warnings
warnings.filterwarnings('ignore')

posLexicon="../resources/positive.txt"
negLexicon="../resources/negative.txt"

positiveProcessedfile="../dataset/positiveProcessed.txt"
negativeProcessedfile="../dataset/negativeProcessed.txt"
neutralProcessedfile="../dataset/neutralProcessed.txt"

positiveFile="../dataset/full_data/posTune.csv"
negativeFile="../dataset/full_data/negTune.csv"
neutralFile="../dataset/full_data/neuTune.csv"

emoticon_file = "../resources/emoticon.txt"
emoticon_dict = microbloggingFeature.create_emoticon_dictionary(emoticon_file)
unicode_emoticon_file = "../resources/emoticon.csv"
unicode_emoticon_dict = microbloggingFeature.create_unicode_emoticon_dictionary(unicode_emoticon_file)

positiveUnigramList=null
negativeUnigramList=null
neutralUnigramList=null
positiveBigramList=null
negativeBigramList=null
neutralBigramList=null
positiveTrigramList=null
negativeTrigramList=null
neutralTrigramList=null


NCpositiveTrigramList=null
NCnegativeTrigramList=null
NCneutralTrigramList=null
NCpositiveFourgramList=null
NCnegativeFourgramList=null
NCneutralFourgramList=null

CHpositiveTrigramList=null
CHnegativeTrigramList=null
CHneutralTrigramList=null
CHpositiveFourgramList=null
CHnegativeFourgramList=null
CHneutralFourgramList=null


def ngramGeneratora():
    global positiveUnigramList,negativeUnigramList,neutralUnigramList,positiveBigramList,neutralTrigramList
    global negativeBigramList,neutralBigramList,positiveTrigramList,positiveTrigramList,negativeTrigramList
    global NCpositiveTrigramList,NCnegativeTrigramList,NCneutralTrigramList
    global NCpositiveFourgramList,NCnegativeFourgramList,NCneutralFourgramList
    global CHpositiveFourgramList,CHpositiveTrigramList,CHneutralFourgramList,CHnegativeFourgramList,CHneutralTrigramList,CHnegativeTrigramList
    print 'creating ngram lists'
    positiveUnigramList=ngramGenerator.ngram(positiveProcessedfile,1,0,0)
    negativeUnigramList=ngramGenerator.ngram(negativeProcessedfile,1,0,0)
    neutralUnigramList=ngramGenerator.ngram(neutralProcessedfile,1,0,0)
    positiveBigramList=ngramGenerator.ngram(positiveProcessedfile,2,0,0)
    negativeBigramList=ngramGenerator.ngram(negativeProcessedfile,2,0,0)
    neutralBigramList=ngramGenerator.ngram(neutralProcessedfile,2,0,0)
    positiveTrigramList=ngramGenerator.ngram(positiveProcessedfile,3,0,0)
    negativeTrigramList=ngramGenerator.ngram(negativeProcessedfile,3,0,0)
    neutralTrigramList=ngramGenerator.ngram(neutralProcessedfile,3,0,0)

    NCpositiveTrigramList = ngramGenerator.ngram(positiveProcessedfile,3,1,0)
    NCnegativeTrigramList = ngramGenerator.ngram(negativeProcessedfile,3,1,0)
    NCneutralTrigramList = ngramGenerator.ngram(neutralProcessedfile,3,1,0)
    NCpositiveFourgramList = ngramGenerator.ngram(positiveProcessedfile,4,1,0)
    NCnegativeFourgramList =ngramGenerator.ngram(negativeProcessedfile,4,1,0)
    NCneutralFourgramList = ngramGenerator.ngram(neutralProcessedfile,4,1,0)

    CHpositiveTrigramList=ngramGenerator.ngram(positiveProcessedfile,3,0,1)
    CHnegativeTrigramList=ngramGenerator.ngram(negativeProcessedfile,3,0,1)
    CHneutralTrigramList=ngramGenerator.ngram(neutralProcessedfile,3,0,1)
    CHpositiveFourgramList=ngramGenerator.ngram(positiveProcessedfile,4,0,1)
    CHnegativeFourgramList=ngramGenerator.ngram(negativeProcessedfile,4,0,1)
    CHneutralFourgramList=ngramGenerator.ngram(neutralProcessedfile,4,0,1)

ngramGeneratora()

def mapTweet(tweet):
    try:
        vector = []
        preprocessed_tweet=preprocess.preProcessTweet(tweet)
        vector.append(lexicon.getLexiconScore(tweet,posLexicon,negLexicon))
        vector.append(microbloggingFeature.emoticon_score(tweet,emoticon_dict))
        vector.append(microbloggingFeature.unicode_emoticon_score(tweet,unicode_emoticon_dict))
        vector.append(writingStyle.uppercasedWordsInTweet(tweet))
        vector.append(writingStyle.exclamationCount(tweet))
        vector.append(writingStyle.questionMarkCount(tweet))
        vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],1))
        vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],2))
        # vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[0],negativeUnigramList[0],neutralUnigramList[0],3))
        vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[1],negativeUnigramList[1],neutralUnigramList[1],1))
        vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[1],negativeUnigramList[1],neutralUnigramList[1],2))
        #vector.extend(ngramGenerator.score(preprocessed_tweet,positiveUnigramList[1],negativeUnigramList[1],neutralUnigramList[1],3))

        #vector.extend(ngramGenerator.score(preprocessed_tweet,NCpositiveTrigramList[0],NCnegativeTrigramList[0],NCneutralTrigramList[0],3))
        #vector.extend(ngramGenerator.score(preprocessed_tweet, NCpositiveFourgramList[0],NCnegativeFourgramList[0], NCneutralFourgramList[0],4))

        # vector.extend(ngramGenerator.score(preprocessed_tweet,CHpositiveTrigramList[0],CHnegativeTrigramList[0],CHneutralTrigramList[0],3))
        # vector.extend(ngramGenerator.score(preprocessed_tweet,CHpositiveFourgramList[0],CHnegativeFourgramList[0],CHnegativeFourgramList[0],4))
        vector.extend(sentiStrength.get_lexiconScore(preprocessed_tweet,1))#get score from senti140
        vector.extend(sentiStrength.get_lexiconScore(preprocessed_tweet,2))#get score from NRC_Hashtag
        vector.append(sentiStrength.getSentiWordScore(preprocessed_tweet))
        vector.append(sentiStrength.getBingLuiScore(preprocessed_tweet))
        return vector
    except:
        print sys.exc_info()

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




