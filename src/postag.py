
import nltk

def posTagString(tweet):
    tag_tweet=""
    tweetWords=""
    tweet =tweet.split()
    for words in tweet:
        tweetWords += words + " "
    taggedTweet = (nltk.pos_tag(tweetWords.replace("_NEG", "").split()))
    for i in range(len(taggedTweet)):
        tag_tweet+= taggedTweet[i][0] + "|"+ taggedTweet[i][1] +" "
    return tag_tweet
