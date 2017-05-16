from mpmath import plot

import lexicon
import microbloggingFeature
import ngramGenerator
import writingStyle
import preprocess
import postag

from sklearn import svm
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing as pr
from sklearn import metrics

positiveUnigram = "../dataset/positiveUnigram.txt"
positiveBigram = "../dataset/positiveBigram.txt"
positiveTrigram = "../dataset/positiveTrigram.txt"
negativeUnigram = "../dataset/negativeUnigram.txt"
negativeBigram = "../dataset/negativeBigram.txt"
negativeTrigram="../dataset/negativeTrigram.txt"
neutralUnigram = "../dataset/neutralUnigram.txt"
neutralBigram = "../dataset/neutralBigram.txt"
neutralTrigram = "../dataset/neutralTrigram.txt"
positivePostaggedUnigram="../dataset/positive_1.txt"
positivePostaggedBigram="../dataset/positive_2.txt"
positivePostaggedTrigram="../dataset/positive_3.txt"
negativePostaggedUnigram="../dataset/negative_1.txt"
negativePostaggedBigram="../dataset/negative_2.txt"
negativePostaggedTrigram="../dataset/negative_3.txt"
neutralPostaggedUnigram="../dataset/neutral_1.txt"
neutralPostaggedBigram="../dataset/neutral_2.txt"
neutralPostaggedTrigram="../dataset/neutral_3.txt"

posLexicon="../resource/positive.txt"
negLexicon="../resource/negative.txt"
emoticonDict=microbloggingFeature.createEmoticonDictionary("../resource/emoticon.txt")

def mapTweet(tweet):
    vector = []
    preprocessed_tweet=preprocess.preProcessTweet(tweet)
    vector.append(lexicon.getLexiconScore(preprocessed_tweet,posLexicon,negLexicon))
    vector.append(microbloggingFeature.emoticonScore(preprocessed_tweet,emoticonDict))
    vector.append(writingStyle.uppercasedWordsInTweet(tweet))
    vector.append(writingStyle.exclamationCount(tweet))
    vector.append(writingStyle.questionMarkCount(tweet))
    vector.extend(ngramGenerator.scoreUnigram(preprocessed_tweet,positiveUnigram,negativeUnigram,neutralUnigram))
    vector.extend(ngramGenerator.scoreBigram(preprocessed_tweet,positiveBigram,negativeBigram,neutralBigram))
    vector.extend(ngramGenerator.scoreTrigram(preprocessed_tweet,positiveTrigram,negativeTrigram,neutralTrigram))
    vector.extend(ngramGenerator.scoreUnigramPostag(str(postag.posTaggedString(preprocessed_tweet)).replace("_NEG",""),positivePostaggedUnigram,negativePostaggedUnigram,neutralPostaggedUnigram))
    vector.extend(ngramGenerator.scoreBigramPostag(str(postag.posTaggedString(preprocessed_tweet)).replace("_NEG",""),positivePostaggedBigram,negativePostaggedBigram,neutralPostaggedBigram))
    vector.extend(ngramGenerator.scoreTrigramPostag(str(postag.posTaggedString(preprocessed_tweet)).replace("_NEG",""),positivePostaggedTrigram,negativePostaggedBigram,neutralPostaggedTrigram))
    return vector

positiveProcessedfile="../dataset/positiveProcessed.txt"
negativeProcessedfile="../dataset/negativeProcessed.txt"
neutralProcessedfile="../dataset/neutralProcessed.txt"

def loadMatrix(posfilename, neufilename, negfilename, poslabel, neulabel, neglabel):
    vectors = []
    labels = []
    kpos = 0
    kneg = 0
    kneu = 0
    print "Loading training dataset..."

    f = open(posfilename, 'r')
    line = f.readline()
    while line:
        try:
            kpos += 1
            z = mapTweet(line)
            vectors.append(z)
            labels.append(float(poslabel))

        except:
            None
        line = f.readline()
    print str(kpos)+"positive lines loaded : "
    f.close()

    f = open(neufilename, "r")
    line = f.readline()
    while line:
        try:
            kneu = kneu + 1
            z = mapTweet(line)
            vectors.append(z)
            labels.append(float(neulabel))

        except:
            None
        line = f.readline()

    f.close()

    f = open(negfilename, 'r')
    line = f.readline()
    while line:
        try:
            kneg = kneg + 1
            z = mapTweet(line)
            vectors.append(z)
            labels.append(float(neglabel))

        except:
            None
        line = f.readline()

    f.close()
    print "Loading done."
    return vectors, labels

# map tweet into a vector
def trainModel(X,Y,knel,c): # relaxation parameter
    clf=svm.SVC(kernel=knel) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    print clf
    plot(clf)
    return clf

X,Y=loadMatrix(positiveProcessedfile,negativeProcessedfile,neutralProcessedfile,'2','-2','0')

# features standardization
X_scaled=pr.scale(np.array(X))
scaler = pr.StandardScaler().fit(X) # to use later for testing data scaler.transform(X)

# features Normalization
X_normalized = pr.normalize(X_scaled, norm='l2') # l2 norm
normalizer = pr.Normalizer().fit(X_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

X=X_normalized
X=X.tolist()
KERNEL_FUNCTION='linear'
C_PARAMETER=0.6

print "Training model with optimized parameters"
MODEL=trainModel(X,Y,KERNEL_FUNCTION,C_PARAMETER)
print "Training done !"

def predict(tweet,model): # test a tweet against a built model
    z=mapTweet(tweet) # mapping
    print z
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return model.predict([z]).tolist()[0] # transform nympy array to list

print predict("I am a  girl",MODEL)