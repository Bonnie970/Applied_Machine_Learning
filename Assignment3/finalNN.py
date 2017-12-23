import numpy as np
import csv

## The sigmoid function
def sigmoid(x,derivative = False):
    if (derivative == True):
        return x*(1.0-x) #This does though only work if input is already from a sigmoid function
    return 1.0/(1+np.exp(-x))

print "Loading data..."
#Loading the data
x = np.loadtxt("train_x.csv",delimiter=",") # load from text
y = np.loadtxt("train_y.csv")
z = np.loadtxt("test_x.csv",delimiter=",")

#converting the y's into vectors of zeros with only one 1 (the class)
ys = np.array ( [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] )
#changing y into only zeros and one 1.
y2 = np.zeros((len(y),40))
for i in xrange(0,np.size(y)):
    for j in xrange(0,40):
        if (y[i] == ys[j]):
            y2[i][j] = 1
            break
y = y2

#partitioning the dataset:
xTrain = x[0:45000]
yTrain = y[0:45000]

xVali = x[45001:49999]
yVali = y[45001:49999]

#The predicting function, that takes in a list of the weight matrices, and the inpu
def predNN(weights,input):
    preds = []
    for i in xrange(0,len(input)):
        outPuts = []
        out0 = sigmoid(np.dot(input[i],weights[0]))
        outPuts.append(out0)
        for l in xrange(1,len(weights)):
            out1 = sigmoid(np.dot(outPuts[l-1],weights[l]))
            outPuts.append(out1)
        preds.append(outPuts[len(weights)-1])
    return preds

#function that generates the list of weight matrices according to specification, and the training data
def genNN(layers,nodes,alpha,iterations,batchSize):
    print "Training..."
    #weigth generation
    np.random.seed(42)
    weights = []
    weightChange = []
    if (layers == 2):
        weights0 = (2*np.random.rand(4096,40) - 1)
        weightChange0 = np.zeros((4096,40))
        weights.append(weights0)
        weightChange.append(weightChange0)
    else:
        weights0 = (2*np.random.rand(4096,nodes) - 1)
        weightChange0 = np.zeros((4096,nodes))
        weights.append(weights0)
        weightChange.append(weightChange0)
        for i in range(0,layers-3):
            weights1 = (2*np.random.rand(nodes,nodes) - 1)
            weightChange0 = np.zeros((nodes,nodes))
            weights.append(weights1)
            weightChange.append(weightChange0)
        weights2 = (2*np.random.rand(nodes,40) - 1)
        weightChange0 = np.zeros((nodes,40))
        weights.append(weights2)
        weightChange.append(weightChange0)

    #run through the iterations
    batchIndex = 0
    for j in xrange(0,iterations):

        #running through data
        for i in xrange(0,len(yTrain)):
            X = xTrain[i][np.newaxis]  #size (1,4096)
            Y = yTrain[i][np.newaxis]  #size (1,40)

            #creating the outPuts of each layer
            outPuts = []
            out0 = sigmoid(np.dot(X,weights[0]))
            outPuts.append(out0)
            for l in xrange(1,layers-1):
                out1 = sigmoid(np.dot(outPuts[l-1],weights[l]))
                outPuts.append(out1)

            #computing the corrections
            corrections = []
            correction0 = sigmoid(outPuts[layers-2],derivative=True)*(Y-outPuts[layers-2])
            corrections.append(correction0)
            for l in xrange(1,layers-1):
                correction1 = sigmoid(outPuts[layers-2-l],True)*np.dot(corrections[l-1],np.transpose(weights[layers-l-1]))
                corrections.append(correction1)

            #adding the weight changes to the weight-change variabel.
            for l in xrange(1,layers-1):
                weightChange[l] += alpha*np.dot(np.transpose(outPuts[l-1]),corrections[layers-l-2])
            weightChange[0] += alpha*np.dot(np.transpose(X),corrections[layers-2])
            batchIndex += 1
            #if the number of examples run through since last weight update is equal to the batch size, update the weights
            if (batchIndex == batchSize):
                for k in xrange(0,len(weights)):
                    weights[k] += weightChange[k]
                    weightChange[k] -= weightChange[k]
                batchIndex = 0
    return weights

A = genNN(3,12,0.05,11,32)
A1 = predNN(A,z)
result = []
for i in xrange(0,len(A1)):
    index = np.argmax(A1[i])
    result.append(ys[index])
#writing the results to a file:
with open("result.csv","wb") as file:
    wr = csv.writer(file)
    wr.writerow(("Id","Label"))
    for i in range(0,len(result)):
        wr.writerow((i+1,result[i]))
