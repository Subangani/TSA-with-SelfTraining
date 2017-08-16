from docutils.parsers import null
import generateVector
import csv
from sklearn import preprocessing as pr
from sklearn import svm

positiveFile="../dataset/full_data/posTrain.csv"
negativeFile="../dataset/full_data/negTrain.csv"
neutralFile="../dataset/full_data/neuTrain.csv"

MODEL=null
model_scaler=null
model_normalizer=null

# train the model
def train_SVM_model(X,Y): # relaxation parameter
    clf=svm.SVC(kernel='rbf', C=0.1,class_weight={2.0: 2.4 ,-2.0: 3.3 }, gamma=0.01) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf

def svm_Model():
    global model_scaler, model_normalizer,MODEL
    X_model, Y_model = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')

    # features standardization
    X_model_scaled = pr.scale(X_model)
    model_scaler = pr.StandardScaler().fit(X_model)  # to use later for testing data scaler.transform(X)

    # features Normalization
    X_model_normalized = pr.normalize(X_model_scaled, norm='l2')  # l2 norm
    model_normalizer = pr.Normalizer().fit(X_model_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test.txt

    X_model = X_model_normalized
    X_model = X_model.tolist()

    #X, Y= generateVector.loadMatrix(positiveTuneFile, neutralTuneFile, negativeTuneFile, '2', '0', '-2')

    # features standardization
    #X_scaled = pr.scale(np.array(X))

    # features Normalization
    #X_normalized = pr.normalize(X_scaled, norm='l2')  # l2 norm

    #X= X_normalized
    #X = X.tolist()
    #KERNEL_FUNCTION, C_PARAMETER = tune.get_Kernel_Cparameter(X_model, Y_model)
    # print "Training model with optimized parameters"
    # KERNEL_FUNCTIONS = ["linear", 'rbf']
    # C = [ 0.01, 0.05, 0.1, 0.5, 1.0]
    # for knel in KERNEL_FUNCTIONS:
    #     for c in C:
    #         print knel + " *** "+ str(c)
    #         MODEL = train_SVM_model(X_model, Y_model,knel, c)
    #         test.txt(MODEL)
    #         print knel + str(c)
    # print "Training done !"
    # return MODEL
    MODEL = train_SVM_model(X_model, Y_model,)
    test(MODEL)

#predict a tweet using the model
def predict(tweet,model): # test.txt a tweet against a built model
    global model_scaler, model_normalizer
    z = generateVector.mapTweet(tweet)  # mapping
    z_scaled = model_scaler.transform(z)
    z = model_normalizer.transform([z_scaled])
    z = z[0].tolist()
    prediction = model.predict([z]).tolist()[0]
    return  prediction # transform nympy array to list

# write labelled  test.txt dataset
def writeTest(filename,model): # function to load test.txt file in the csv format : sentiment,tweet
    f = open(filename, "r")
    reader = csv.reader(f)
    fo=open(filename+".svm_result","w")
    fo.write("old,tweet,new\n")
    for line in reader:
        tweet = line[2]
        s = line[1]
        nl=predict(tweet,model)
        fo.write(r'"'+str(s)+r'","'+tweet+r'","'+str(nl)+r'"'+"\n")
    f.close()
    fo.close()
    print "labelled testdataset is stores in : "+str(filename)+".svm_result"

def test(model):
    print "Loading test.txt data..."
    writeTest('../dataset/test.csv', model)
    getAccuracyPrecision()

def get_divided_value(numerator, denominator):
    if denominator == 0:
        return 0.0
    else:
        result = numerator/(denominator * 1.0)
        return round(result, 4)


def getAccuracyPrecision():
    TP = TN = TNeu = FP_N = FP_Neu = FN_P = FN_Neu = FNeu_P = FNeu_N = 0
    try:
        testedFile="../dataset/test.csv.svm_result"
        f0=open(testedFile,"r")
        line = f0.readline()
        readers = csv.reader(f0)
        dic = {'positive': 2.0, 'negative': -2.0, 'neutral': 0.0}
        for line in readers:
            new = str(line[2])
            old = str(dic.get(line[0]))
            if old == new:
                if new == "2.0":
                    TP += 1
                elif new == "-2.0":
                    TN += 1
                elif new == "0.0":
                    TNeu += 1
            else:
                if new == "2.0" and old == "-2.0":
                    FP_N += 1
                elif new == "2.0" and old == "0.0":
                    FP_Neu += 1
                elif new == "-2.0" and old == "2.0":
                    FN_P += 1
                elif new == "-2.0" and old == "0.0":
                    FN_Neu += 1
                elif new == "0.0" and old == "2.0":
                    FNeu_P += 1
                elif new == "0.0" and old == "-2.0":
                    FNeu_N += 1
        accuracy = get_divided_value((TP + TN + TNeu),(TP + TN + TNeu + FP_N + FP_Neu + FN_P +FN_Neu + FNeu_P + FNeu_N))
        pre_p = get_divided_value(TP, (FP_N + FP_Neu + TP))
        pre_n = get_divided_value(TN, (FN_P + FN_Neu + TN))
        pre_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
        re_p = get_divided_value(TP, (FN_P + FNeu_P + TP))
        re_n = get_divided_value(TN, (FP_N + FNeu_N + TN))
        re_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
        f_score_p = 2 * get_divided_value((re_p * pre_p),(re_p + pre_p))
        f_score_n = 2 * get_divided_value((re_n * pre_n),(re_n + pre_n))
        f_score_average = round((f_score_p + f_score_n)/2,4)
        print accuracy, pre_p, pre_n, pre_neu, re_p, re_n, re_neu, f_score_p, f_score_n,f_score_average
    except IOError:
        print IOError.filename



svm_Model()

# acc,precision_pos,precision_neg,precision_neu,recall_pos,recall_neg,recall_neu
# 0.51326929339 0.413522012579 0.529331976941 0.557108042242 0.553684210526 0.393001007049 0.577564426478
# F_core_pos,F_core_neg,F_core_neu
# 0.473447344734 0.45109088282 0.567151835925
