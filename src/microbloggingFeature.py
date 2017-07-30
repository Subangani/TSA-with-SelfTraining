from __future__ import division
import csv


def create_emoticon_dictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative': -0.5, 'Extremely-Negative': -1.0,
                  'Neutral': 0.0}
    emo_score_list = {}
    fi = open(filename, "r")
    l = fi.readline()
    while l:
        l = l.replace("\xc2\xa0", " ")
        li = l.split(" ")
        l2 = li[:-1]
        l2.append(li[len(li) - 1].split("\t")[0])
        sentiment = li[len(li) - 1].split("\t")[1][:-2]
        score = emo_scores[sentiment]
        l2.append(score)
        for i in range(0, len(l2) - 1):
            emo_score_list[l2[i]] = l2[len(l2) - 1]
        l = fi.readline()
    return emo_score_list


def create_unicode_emoticon_dictionary(filename):
    emo_score_list = {}
    with open(filename, "r") as unlabeled_file:
        reader = csv.reader(unlabeled_file)
        for line in reader:
            emo_score_list.update({str(line[0]):float(line[4])})
    return emo_score_list


def emoticon_score(tweet, d):
    s = 0.0
    l = tweet.split(" ")
    nbr = 0
    for i in range(0, len(l)):
        if l[i] in d.keys():
            nbr = nbr + 1
            s = s + d[l[i]]
    if nbr != 0:
        s = s / nbr
    return s


def unicode_emoticon_score(tweet,d):
    s = 0.0
    nbr = 0
    tweet = tweet.split();
    for i in range(len(tweet)):
        old = tweet[i]
        new = old.replace("\U000","0x")
        if new in d.keys() :
            nbr = nbr + 1
            s = s + d[new]
    if nbr != 0:
        s = s/nbr
    return s


def hastag_dict(filename):
    has_score = {'positive': 1, 'negative': -1}
    has_score_list = {}
    fi = open(filename, "r")
    l = fi.readline()
    while l:
        try:
            li = l.split("\t")
            senti = li[1]
            score = has_score[senti[:-1]]
            has_score_list[li[0]] = score
        except IndexError:
            print "Error"
        l = fi.readline()
    return has_score_list


def hashtag_words(tweet, dict):
    l = tweet.split()
    result = []
    for w in l:
        if w[0] == '#':
            for word in dict.keys:
                if word in w:
                    score = score + word.key
            result.append(w, score)
    return result

# d = create_unicode_emoticon_dictionary('../resource/emoticon.csv')
# tweet = '\U0001f602 \U0001f601 \U0001f622 '
# print unicode_emoticon_score(tweet,d)