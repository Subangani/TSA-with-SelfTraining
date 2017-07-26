from __future__ import division

def createEmoticonDictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative':-0.5,'Extremely-Negative': -1.0,'Neutral': 0.0}
    emo_score_list={}
    fi = open(filename,"r")
    l=fi.readline()
    while l:

        l=l.replace("\xc2\xa0"," ")
        li=l.split(" ")
        l2=li[:-1]
        l2.append(li[len(li)-1].split("\t")[0])
        sentiment=li[len(li)-1].split("\t")[1][:-1]
        score=emo_scores[sentiment]
        l2.append(score)
        for i in range(0,len(l2)-1):
            emo_score_list[l2[i]]=l2[len(l2)-1]
        l=fi.readline()
    return emo_score_list

def emoticonScore(tweet,d):
    s=0.0;
    l=tweet.split(" ")
    nbr=0;
    for i in range(0,len(l)):
        if l[i] in d.keys():
            nbr=nbr+1
            s=s+d[l[i]]
    if (nbr!=0):
        s=s/nbr
    return s

def hashtagWords(tweet):
    l = tweet.split()
    result = []
    for w in l:
        if w[0] == '#':
            result.append(w)
    return result

def createTrainingSet(fileName):
    with open(fileName) as fp:
        f1 = open(fileName + ".trainingset", "w")
        f2= open(fileName+".testset", "w")
        count=0
        for line in fp:
            if (count %4 ==0):
              f2.write(line)
            else:
                f1.write(line)
            count+=1


# f1="../dataset/positive.csv"
# createTrainingSet(f1)
#
# f1="../dataset/neutral.csv"
# createTrainingSet(f1)
#
# f1="../dataset/negative.csv"
# createTrainingSet(f1)

# 0.575757575758 0.625 0.384615384615