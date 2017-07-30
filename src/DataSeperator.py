
import csv
def seperate():
    pos="../dataset/full_data/positive.csv"
    neg="../dataset/full_data/negative.csv"
    neu="../dataset/full_data/neutral.csv"

    posTrain = "../dataset/posTrain.csv"
    negTrain = "../dataset/negTrain.csv"
    neuTrain = "../dataset/neuTrain.csv"

    posSelf = "../dataset/posSelf.csv"
    negSelf = "../dataset/negSelf.csv"
    neuSelf = "../dataset/neuSelf.csv"

    f0=open(pos,'r')
    f1=open(neg,'r')
    f2=open(neu,'r')

    #f3=open(posTrain,"w")
    #f4=open(negTrain,"w")
    f5=open(neuTrain,"w")

    #f6=open(posSelf,"w")
    #f7=open(negSelf,"w")
    f8=open(neuSelf,"w")
    reader = csv.reader(f2)
    x0 = 0
    for row in reader:
        x0+=1
        if(x0%2):
            f5.write(str(row)+"\n")
        else:
            f8.write(str(row)+"\n")


seperate()