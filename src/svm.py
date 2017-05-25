import lexicon
import microbloggingFeature
import ngramGenerator
import writingStyle
import preprocess
import postag

from sklearn import svm
import numpy as np
from sklearn import preprocessing as pr
import csv
import warnings
warnings.filterwarnings('ignore')

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
    print str(kneu) + "neutral lines loaded : "
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
    print str(kneg) + "negative lines loaded : "
    print "Loading done."
    return vectors, labels

# map tweet into a vector
def trainModel(X,Y,knel,c): # relaxation parameter
    clf=svm.SVC(kernel=knel, C=c) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    print clf

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
C_PARAMETER=1.0

print "Training model with optimized parameters"
MODEL=trainModel(X,Y,KERNEL_FUNCTION,C_PARAMETER)
print "Training done !"

def predict(tweet,model): # test a tweet against a built model
    z=mapTweet(tweet) # mapping
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return model.predict([z]).tolist()[0] # transform nympy array to list


def predictFile(filename, svm_model):  # function to load test file in the csv format : sentiment,tweet
    f = open(filename, 'r')
    fo = open(filename + ".result", 'w')
    fo.write('"auto label","tweet","anouar label","ziany label"')  # header
    line = f.readline()
    while line:
        tweet = line[:-1]
        nl = predict(tweet, svm_model)
        fo.write(r'"' + str(nl) + r'","' + tweet + '"\n')
        line = f.readline()
    f.close()
    fo.close()
    print "Tweets are classified . The result is in " + filename + ".result"


def loadTest(filename): # function to load test file in the csv format : sentiment,tweet
    labels=[]
    vectors=[]
    f0 = open(filename, "r")
    reader = csv.reader(f0)
    for row in reader:
        tweet = row[1]
        s=row[0]
        z=mapTweet(tweet)

        z_scaled=scaler.transform(z)
        z=normalizer.transform([z_scaled])
        z=z[0].tolist()
        vectors.append(z)
        labels.append(s)
    f0.close()
    return vectors,labels

# write labelled  test dataset
def writeTest(filename,model): # function to load test file in the csv format : sentiment,tweet
    f = open(filename, "r")
    reader = csv.reader(f)
    fo=open(filename+".svm_result","w")
    fo.write("old,tweet,new\n")
    for line in reader:
        tweet = line[1]
        s = line[0]
        nl=predict(tweet,model)
        fo.write(r'"'+str(s)+r'","'+tweet+r'","'+str(nl)+r'"'+"\n")
#        print str(kneg)+"negative lines loaded"
    f.close()
    fo.close()
    print "labelled test dataset is stores in : "+str(filename)+".svm_result"

def getAccuracyPrecision():
    testedFile="../dataset/test.csv.svm_result"
    f0=open(testedFile,"r")
    line = f0.readline() #pass first line
    tweet=f0.readline()
    TP=0
    TN=0
    TNeu=0
    FP=0
    FN=0
    FNeu=0
    while tweet:
        tweetResult= tweet.replace('"','').split(",")
        if ((tweetResult[0]=="positive") & (str(tweetResult[2])=="2.0\n")):
            TP+=1
        if ((tweetResult[0]=="negative") & (tweetResult[2]=="-2.0\n")):
            TN+=1
        if ((tweetResult[0]=="neutral") & (tweetResult[2]=="0.0\n")):
            TNeu+=1
        if (((tweetResult[0]=="negative") | (tweetResult[0]=="neutral")) & (tweetResult[2]=="2.0\n")):
            FP+=1
        if (((tweetResult[0]=="positive")| (tweetResult[0]=="neutral")) & (tweetResult[2]=="-2.0\n")):
            FN+=1
        if (((tweetResult[0]=="positive")| (tweetResult[0]=="negative")) & (tweetResult[2]=="0.0\n")):
            FNeu+=1
        tweet=f0.readline()
    acc=(TP+TN+TNeu)/((TP+TN+TNeu+FP+FN+FNeu)*1.0)
    precision=(TP/((FP+TP)*1.0))
    recall=TP/(TP+FN)
    print acc,precision
    return acc, precision

# uncomment to classify test dataset
print "Loading test data..."
#V, L = loadTest('../dataset/test.csv')
writeTest('../dataset/test.csv', MODEL)
# writ labelled test dataset
accuracy,precision = getAccuracyPrecision()
