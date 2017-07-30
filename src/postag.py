import nltk

def pos_tag_string(tweet):
    adjective_list = ["JJ", "JJR", "JJS"]
    verb_list = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    adverb_list = ["RB", "RBR", "RBS", "WRB"]
    selection_list= adjective_list + verb_list + adverb_list
    tag_tweet = ""
    tweet_words = ""
    tweet = tweet.split()
    for words in tweet:
        tweet_words += words + " "
    tagged_tweet = (nltk.pos_tag(tweet_words.replace("_NEG", "").split()))
    for i in range(len(tagged_tweet)):
        if tagged_tweet[i][1] in selection_list :
            tag_tweet += tagged_tweet[i][0] + "|"+ tagged_tweet[i][1] +" "
    return tag_tweet
