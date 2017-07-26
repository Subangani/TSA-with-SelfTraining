from docutils.nodes import classifier

import generateVector

positiveProcessedfile="../dataset/positiveProcessed.txt"
negativeProcessedfile="../dataset/negativeProcessed.txt"
neutralProcessedfile="../dataset/neutralProcessed.txt"

#TO BE IMPLEMENTED CORRECTLY
def selfTraining():
    f0=open("../dataset/unlabeled.txt","r")
    tweet=f0.readline()
    global MODEL
    while tweet:
        for i in range(1, 20000, 1000):
            f1 = open(positiveProcessedfile, "a")
            f2 = open(negativeProcessedfile, "a")
            f3 = open(neutralProcessedfile, 'a')
            for j in range(i, i + 1000, 1):
                tweet = f0.readline()
                sentiment = generateVector.predict(tweet, MODEL)
                if (sentiment == float(2.0)):
                    f1.write(tweet)
                if (sentiment == float(0.0)):
                    f3.write(tweet)
                if (sentiment == float(-2.0)):
                    f2.write(tweet)
            f1.close()
            f2.close()
            f3.close()
            generateVector.ngramGeneratora()
            generateVector.svm_Model()
            generateVector.test()

    print "added"


#selfTraining()