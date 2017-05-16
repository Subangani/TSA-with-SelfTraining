import nltk

def posTag(tweet):
    tweetWords=""
    for i in range(len(tuple(tweet))):
        tweetWords+=tweet[i] +" "
    taggedTweet=(nltk.pos_tag(tweetWords.replace("_NEG","").split()))
    return taggedTweet

def posTaggedString(tweet):
    tweetWords = ""
    for i in range(len(tweet.split(" "))):
        tweetWords += tweet.split(" ")[i] + " "
    taggedTweet = (nltk.pos_tag(tweetWords.replace("_NEG", "").split()))
    taggedtweetStr = (str(taggedTweet)).replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace("(","").replace(")", "").split(",")

    return taggedtweetStr