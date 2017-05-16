from __future__ import division

import json


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
        score=emo_scores.get("Positive")
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

