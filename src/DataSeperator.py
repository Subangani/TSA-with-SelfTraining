
import csv
def seperate():
    pos="../dataset/full_data/positive.csv"
    neg="../dataset/full_data/negative.csv"
    neu="../dataset/full_data/neutral.csv"

    posTrain = "../dataset/full_data/posTrain.csv"
    negTrain = "../dataset/full_data/negTrain.csv"
    neuTrain = "../dataset/full_data/neuTrain.csv"

    posTune = "../dataset/full_data/posTune.csv"
    negTune = "../dataset/full_data/negTune.csv"
    neuTune = "../dataset/full_data/neuTune.csv"

    f0=open(pos,'r')
    f1=open(neg,'r')
    f2=open(neu,'r')

    f3=open(posTrain,"w")
    #f4=open(negTrain,"w")
    #f5=open(neuTrain,"w")

    f6=open(posTune,"w")
    #f7=open(negTune,"w")
    #f8=open(neuTune,"w")
    line=f0.readline()
    x0 = 0
    while line:
        x0+=1
        if(x0%3==1 or x0%3==2):
            f3.write(str(line))
        else:
            f6.write(str(line))
        line=f0.readline()
seperate()

