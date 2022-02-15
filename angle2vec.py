from scipy.io import mmread
from scipy.sparse import csr_matrix
import sys
import networkx as nx
import random, math
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

def scale(x):
    bound = 6
    if x > bound:
        return bound
    elif x < -bound:
        return -bound
    return x


def conceptTest_f2v(graph, iterations, dim):
    X = [[random.random() for j in range(dim)] for i in range(graph.shape[0])]
    print(len(X), len(X[0]))
    STEP = 0.02 # step length is the learning rate
    TH = 9999999
    EPS = 0.00001
    for it in range(iterations):
        for u in range(graph.shape[0]):
            fnx = [0 for j in range(dim)]
            normx = 0 #[0 for j in range(dim)]
            normneigh = 0
            Nu = graph.indices[graph.indptr[u]:graph.indptr[u+1]]
            for j in Nu:
                attract = 0
                for d in range(dim):
                    fnx[d] = X[u][d] - X[j][d]
                    attract += fnx[d] * fnx[d]
                d1 = -2.0 / (1.0 + attract)
                for d in range(dim):
                    fnx[d] = scale(fnx[d] * d1)
                    X[u][d] += STEP * fnx[d]
            for j in range(15):
                k = random.randint(0, graph.shape[0]-1)
                repulse = 0
                for d in range(dim):
                    fnx[d] = X[u][d] - X[k][d]
                    repulse += fnx[d] * fnx[d]
                d2 = 2.0 / ((repulse + EPS) * (1.0 + repulse))
                for d in range(dim):
                    fnx[d] = scale(fnx[d] * d2)
                    X[u][d] += STEP * fnx[d]
    return np.array(X)   

from sklearn.multiclass import OneVsRestClassifier
class MyClass(OneVsRestClassifier):
    def prediction(self, X, nclasses):
        ps = np.asarray(super(MyClass, self).predict_proba(X))
        predlabels = []
        for i, k in enumerate(nclasses):
            ps_ = ps[i, :]
            labels = self.classes_[ps_.argsort()[-k:]].tolist()
            predlabels.append(labels)
        return predlabels

def makeNodeClassificationData(X, truthlabelsfile):
    Xd = [[] for i in range(len(X))]
    Yd = [[] for i in range(len(X))]
    distinctlabels = set()
    lfile = open(truthlabelsfile)
    for line in lfile.readlines():
        tokens = line.strip().split()
        node = int(tokens[0])-1
        label = int(tokens[1])
        Xd[node] = X[node]
        Yd[node].append(label)
        distinctlabels.add(label)
    lfile.close()
    Xd = [row for row in Xd if len(row) > 0]
    Yd = [row for row in Yd if len(row) > 0]
    return np.array(Xd), np.array(Yd), len(distinctlabels)

def train_test(Xt, Yt, num_distinct_labels):
    indices = np.array(range(len(Yt)))
    onehotencode = MultiLabelBinarizer(range(num_distinct_labels))
    trainfrac = [0.05, 0.10, 0.15]
    for tf in trainfrac:
        np.random.shuffle(indices)
        CV = int(len(Yt) * tf)
        trainX = Xt[indices[0:CV]]
        testX = Xt[indices[CV:]]
        trainY = Yt[indices[0:CV]]
        trainY = onehotencode.fit_transform(trainY)
        testY = Yt[indices[CV:]]
        modelLR = MyClass(LogisticRegression(random_state=0)).fit(trainX, trainY)
        ncs = [len(x) for x in testY]
        predictedY = modelLR.prediction(testX, ncs)
        testY = onehotencode.fit_transform(testY)
        f1macro = f1_score(onehotencode.fit_transform(predictedY), testY, average='macro')
        f1micro = f1_score(onehotencode.fit_transform(predictedY), testY, average='micro')
        print("Multilabel-classification:", tf, "F1-macro:", f1macro, "F1-micro:",f1micro)

if __name__ == "__main__":
    input_file = sys.argv[1]
    label_file = sys.argv[2]
    G = mmread(input_file)
    G_arr = G.toarray()
    G_csr = csr_matrix(G_arr)
    if len(sys.argv) > 3:
        epoch = int(sys.argv[3])
    else:
        epoch = 100
    print("graph_file:", input_file, "label_file:", label_file, "epoch:", epoch)
    X = conceptTest_f2v(G_csr, epoch, 32) # 32-dimensional embedding
    print("Generated embedding!", X.shape)
    Xt, Yt, num_distinct_labels = makeNodeClassificationData(X, label_file)
    train_test(Xt, Yt, num_distinct_labels) 
