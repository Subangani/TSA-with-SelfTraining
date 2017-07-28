
import postag
import csv

def dict(words,gram):
    """
    This is to obtain the dict of word of particular line
    :param words: 
    :param gram: 
    :return: give a dict of word(s) with proper format based ngram values such as 1,2,3
    """
    tempdict={}
    for i in range(len(words)+1 - gram):
        if not words[i:i + gram] is "":
            temp = ""
            if (gram == 1):
                temp = words[i]
            elif (gram == 2):
                temp = words[i] + " " + words[i + 1]
            elif (gram == 3):
                temp = words[i] + " " + words[i + 1] + " " + words[i + 2]
            local_temp_value =tempdict.get(temp)
            if(local_temp_value is None):
                tempdict.update({temp:1})
            else:
                tempdict.update({temp:local_temp_value+1})
    return tempdict

def ngram(file,gram):
    """
    Inputs files and produce ngram values based on gram. eg: gram = 1,2,3
    :param file: 
    :param gram: 
    :return: frequency of ngram, and frequency of postag ngram as dictionary
    """
    is_success=False
    freq_list={}
    postag_freq_list={}
    with open(file,"r") as main:
        reader = csv.reader(main)
        for line in reader:
            try:
                words=line[0].split()
                word_dict=dict(words,gram)
                freq_list,is_success=dictUpdate(freq_list,word_dict)
                postags=postag.posTagString(line[0]).split()
                postag_dict=dict(postags,gram)
                postag_freq_list,is_success=dictUpdate(postag_freq_list,postag_dict)
            except IndexError:
                print "Error"
    is_success=True

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
    return  count

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
    dictofGrams.update(dict(tweet_list,ngram))
    for element in dictofGrams.keys():
        posCount = float(get_count(element,p))
        negCount = float(get_count(element,n))
        neuCount = float(get_count(element,ne))
        totalCount = posCount + negCount + neuCount
        if (totalCount != 0):
            pos += posCount / totalCount
            neg += negCount / totalCount
            neu += neuCount / totalCount
    return [pos,neg,neu]



def dictUpdate(original,temp):
    """
    This will update original dictionary key, and values by comparing with temp values
    :param original:
    :param temp:
    :return: original updated dictionary and a success statement
    """
    is_success=False
    for key in temp.keys():
        global_key_value=original.get(key)
        local_key_value=temp.get(key)
        if(global_key_value is None):
            original.update({key:local_key_value})
        else:
            original.update({key:local_key_value+global_key_value})
    is_success=True
    return original,is_success



