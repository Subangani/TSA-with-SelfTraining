import nltk

def pos_tag_string(tweet):
    adjective_list = ["JJ", "JJR", "JJS"]
    verb_list = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    adverb_list = ["RB", "RBR", "RBS", "WRB"]
    noun_list = ['NN', 'NNS', 'NNP', 'NNPS']
    tag_tweet = ""
    tweet_words = ""
    tweet = tweet.split()
    for words in tweet:
        tweet_words += words + " "
    tagged_tweet = (nltk.pos_tag(tweet_words.replace("_NEG", "").split()))
    for i in range(len(tagged_tweet)):
        if tagged_tweet[i][1] in adjective_list:
            tag_tweet += tagged_tweet[i][0] + "|"+ "adj" +" "
        elif tagged_tweet[i][1] in verb_list:
            tag_tweet += tagged_tweet[i][0] + "|"+ "verb" +" "
        elif tagged_tweet[i][1] in adverb_list:
            tag_tweet += tagged_tweet[i][0] + "|"+ "adv" +" "
        elif tagged_tweet[i][1] in noun_list:
            tag_tweet += tagged_tweet[i][0] + "|"+ "noun" +" "
    return tag_tweet
