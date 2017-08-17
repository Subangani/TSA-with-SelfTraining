import postag
import csv
from nltk.stem.snowball import SnowballStemmer


def dict(words,gram,NC,CH):
    """
    This is to obtain the dict of word of particular line
    :param words: 
    :param gram: 
    :return: give a dict of word(s) with proper format based ngram values such as 1,2,3
    """
    tempdict={}
    if (CH):
        tempdict=characterNgram(words,gram)
    else:
        for i in range(len(words)+1 - gram):
            if not words[i:i + gram] is "":
                temp = ""
                if (gram == 1):
                    temp = words[i]
                elif (gram == 2):
                    temp = words[i] + " " + words[i + 1]
                elif (gram == 3):
                    if(NC):
                        temp = words[i] + " " +"*" + " " + words[i + 2]
                    else:
                        temp = words[i] + " " + words[i + 1] + " " + words[i + 2]
                elif (gram == 4):
                    if(NC):
                        temp = words[i] + " " +"*" + " " +"*"+" "+ words[i + 3]
                    else:
                        temp = words[i] + " " + words[i + 1] + " " + words[i + 2]+ " " + words[i + 3]
                local_temp_value =tempdict.get(temp)
                if(local_temp_value is None):
                    tempdict.update({temp:1})
                else:
                    tempdict.update({temp:local_temp_value+1})
    return tempdict

def stemTweets(tweet):
    stemmer = SnowballStemmer("english")
    splittedtTweet=tweet.split()
    tweet=[stemmer.stem(stemmedTweet.replace("_NEG","")) for stemmedTweet in splittedtTweet]
    tweet=(' '.join(tweet))
    return tweet

def ngram(file,gram,NC,CH):
    """
    Inputs files and produce ngram values based on gram. eg: gram = 1,2,3
    :param file: 
    :param gram: 
    :return: frequency of ngram, and frequency of postag ngram as dictionary
    """
    freq_list={}
    characterList={}
    postag_freq_list={}
    with open(file,"r") as main:
        reader = csv.reader(main)
        for line in reader:
            try:
                words = line[0].split()
                word_dict = dict(words, gram,NC,CH)
                freq_list,is_success=dictUpdate(freq_list,word_dict)
                postags=postag.pos_tag_string(line[0]).split()
                postag_dict=dict(postags,gram,0,0)
                postag_freq_list,is_success=dictUpdate(postag_freq_list,postag_dict)
                # chara_dict=dict(line[0],gram,NC,CH)
                # characterList,is_success=dictUpdate(characterList,chara_dict)
            except IndexError:
                print "Error"
    return freq_list,postag_freq_list

def get_count(gram,pol):
    """
    This will count the positive,negative, and neutral count based on relevant dictionary present
    :param gram: 
    :param pol: 
    :return: return the count of particular ngram
    """
    try:
        count = 0.0
        count = float(pol.get(gram))
    except:
        TypeError
    return count

def score(tweet,p,n,ne,ngram):
    """
    This will find individual score of each word with respect to its polarity
    :param tweet: 
    :param p: 
    :param n: 
    :param ne: 
    :param ngram: 
    :return: return positive, negative, and neutral score
    """
    pos=0
    neg=0
    neu=0
    dictofGrams={}
    tweet_list=tweet.split()
    dictofGrams.update(dict(tweet_list,ngram,1,0))
    for element in dictofGrams.keys():
        posCount = float(get_count(element,p))
        negCount = float(get_count(element,n))
        neuCount = float(get_count(element,ne))
        totalCount = posCount + negCount + neuCount
        if (totalCount!= 0):
            pos += posCount / totalCount
            neg += negCount / totalCount
            neu += neuCount / totalCount
    return [pos,neg,neu]


def dictUpdate(original, temp):
    """
    This will update original dictionary key, and values by comparing with temp values
    :param original:
    :param temp:
    :return: original updated dictionary and a success statement
    """
    is_success = False
    result = {}
    for key in temp.keys():
        global_key_value = original.get(key)
        local_key_value = temp.get(key)
        if key not in original.keys():
            result.update({key:local_key_value})
        else:
            result.update({key: local_key_value + global_key_value})
    for key in original.keys():
        if key not in temp.keys():
            result.update({key:original.get(key)})
    is_success = True
    return result,is_success


def characterNgram(tweet,gram):
    for word in tweet:
        characters=list(word)
        tempdict = {}
        if (characters.__len__()>3):
            for i in range(len(characters) + 1 - gram):
                if not characters[i:i + gram] is "":
                    temp = ""
                    if (gram == 3):
                        temp = characters[i] + " " + characters[i + 1] + " " + characters[i + 2]
                    local_temp_value = tempdict.get(temp)
                    if (local_temp_value is None):
                        tempdict.update({temp: 1})
                    else:
                        tempdict.update({temp: local_temp_value + 1})
        if (characters.__len__()>4):
            for i in range(len(characters) + 1 - gram):
                if not characters[i:i + gram] is "":
                    temp = ""
                    if (gram == 4):
                        temp = characters[i] + " " + characters[i + 1] + " " + characters[i + 2] + " " + characters[i + 3]
                    local_temp_value = tempdict.get(temp)
                    if (local_temp_value is None):
                        tempdict.update({temp: 1})
                    else:
                        tempdict.update({temp: local_temp_value + 1})
    return tempdict


