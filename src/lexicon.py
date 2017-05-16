from __future__ import division
import re


def loadWords(filename):
    f = open(filename, 'r')
    words = {}
    line = f.readline()
    lines = line.split(",");
    return lines


def getLexiconScore(tweet, pos, neg):
    score = 0;
    posW = loadWords(pos);
    negW = loadWords(neg);
    for w in tweet.split():
        if w in posW:
            score += 1;

        if w in negW:
            score -= 1;

        if w.endswith("_NEG"):
            if len(w) > 4:
                if (w[0:len(w) - 4]) in posW:
                    score -= 1;
                if (w[0:len(w) - 4]) in negW:
                    score += 1;
    return score;




# input tweet after removing punctuations
# 1 tweet