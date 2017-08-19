from docutils.parsers import null
import generateVector
import csv
import time
import general as gen
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
    clf=svm.SVC(kernel='rbf', C=1.0,class_weight={2.0: 2.4 ,-2.0: 3.3 }, gamma=0.01) # linear, poly, rbf, sigmoid, precomputed , see doc
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
    gen.get_scores()


timelist=[]
timelist.append(time.time())
svm_Model()
timelist.append(time.time())
print "Time taken to process is " + str(gen.temp_difference_cal(timelist))

# acc,precision_pos,precision_neg,precision_neu,recall_pos,recall_neg,recall_neu
# 0.514083360469 0.41407736208 0.530328702135 0.557371743 0.549894736842 0.394008056395 0.580090955028
# F_core_pos,F_core_neg,F_core_neu
# 0.472418158799 0.452116134624 0.568504456916
