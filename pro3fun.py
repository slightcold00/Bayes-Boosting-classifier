import numpy as np
from labfuns import *
from scipy import misc
from imp import reload
import random

def mlParams(X, y, W):
    assert(X.shape[0]==y.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(y)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    sigma_mm = np.zeros((Nclasses,Ndims))
    p = np.zeros((Nclasses,1))

    for k in range(Nclasses):
        idx = np.where(y==k)[0]
        Xc = X[idx,:] # Get the x for the k label. Vectors are rows.
        Wc = W[idx,:]
        #calculate mu
        WXc = np.zeros(Xc.shape)
        WXc_2 = np.zeros(Xc.shape)
        for j in range(Xc.shape[0]):
            WXc[j,:] = Wc[j] * Xc[j,:]
        
        mu[k,:] = WXc.sum(axis=0)/np.sum(Wc)
        #calculate sigema
        for j in range(Xc.shape[0]):
            WXc_2[j,:] = Wc[j] * ((Xc[j,:] - mu[k])**2)
        sigma_mm[k,:] = WXc_2.sum(axis=0)/np.sum(Wc)
        sigma[k,:,:] =  np.diag(sigma_mm[k,:])
        #calculate prior
        p[k] = np.sum(Wc)
    # ==========================

    return p, mu, sigma

X,y = genBlobs()
p, mu, sigma= mlParams(X, y, W = None)
print(mu,sigma,p)

def classifyBayes(X, p, mu, sigma):
    Npts,Ndims = np.shape(X)
    Nclasses = np.shape(mu)[0]   
    classifier = np.zeros((Npts,Nclasses))
    h = np.zeros((1,Npts))
    for k in range(Nclasses):
        sigdet_k = np.linalg.det(sigma[k,:,:])
        for i in range(Npts):
            classifier[i,k] = (-1/2) * np.log(sigdet_k) - \
            (1/2)* np.dot(np.dot(X[i,:]-mu[k], np.diag(1/np.diag(sigma[k,:,:]))), (X[i,:]-mu[k]).reshape(-1,1))\
             + np.log(p[k])
    for j in range(Npts):
        z = classifier[j,0]
        for k in range(Nclasses):
            if classifier[j,k] >= z:
                z = classifier[j,k]
                h[0,j] = k
    return h

class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None ):
        rtn = BayesClassifier()
        rtn.prior = mlParams(X, labels, W)[0]
        rtn.mu = mlParams(X, labels, W)[1]
        rtn.sigma = mlParams(X, labels, W)[2]
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)

#Classifier = BayesClassifier()
B = testClassifier(BayesClassifier(), dataset='iris', split=0.7)
print(B)
#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        h = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        errc = np.zeros((Npts,1))
        for j in range(Npts):
            if h[0,j] == labels[j]:
                errc[j,0] = 0
            else:
                errc[j,0] = wCur[j,0]
        err = np.sum(errc)

        if err == 0:
            alphas.append(0)
        else:
            alphat = (1/2) * (np.log(1-err)-np.log(err))   
            # alphas.append(alpha) # you will need to append the new alpha
            alphas.append(alphat)
            for k in range(Npts):
                if h[0,k] == labels[k]:
                    wCur[k,0] = wCur[k,0] * np.exp(-alphat)
                else:
                    wCur[k,0] = wCur[k,0] * np.exp(alphat)
            wCur = wCur/np.sum(wCur)

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        votes_t = np.zeros((Npts,Nclasses,Ncomps))
        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for i in range(Npts):
            for j in range(Nclasses):
                for t in range(Ncomps):
                    h = classifiers[t].classify(X)
                    if h[0,i] == j:
                        votes_t[i,j,t] = alphas[t]
                    else:
                        votes_t[i,j,t] = 0
                votes[i,j] = np.sum(votes_t[i,j,t])      
        # ==========================
        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)


#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)


#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])