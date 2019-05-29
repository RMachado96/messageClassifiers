import pickle
import math
import pandas as pd
import numpy as np
import datetime
import warnings
import sklearn
from sklearn.svm import SVC
import time
sklearn.warnings.filterwarnings("ignore")

def makeSplits(currList):
    #create three splits for validation, training and testing
    trainSplit = 0
    testSplit = 0
    validationSplit = 0

    totalLines = len(currList)
    firstSplit = round(totalLines*0.7)

    trainSplit = currList[0:firstSplit]

    secondSplit = firstSplit + round(totalLines*0.15)

    testSplit = currList[firstSplit : secondSplit]

    validationSplit = currList[secondSplit : totalLines]

    return trainSplit,testSplit,validationSplit

def createWordArray(currList):
    #creates an array of the following format ['target','word1','word2',...,'wordx]
    wordArray = []
    wordArray.append('target')
    for i in currList:
        for j in i:
            j = j.lower()
            if j not in wordArray:
                if j != 'spam' and j != 'ham':
                    wordArray.append(j)

    return wordArray                

def createWordArrayNoTarget(currList):
    #similiar to createWordArray but without the first value 'target'
    wordArrayNoTarget = []
    for i in currList:
        for j in i:
            j = j.lower()
            if j not in wordArrayNoTarget:
                if j != 'spam' and j != 'ham':
                    wordArrayNoTarget.append(j)

    return wordArrayNoTarget                

def createMatrixWordCount(currList,wordArray):
    #goes through the chosen list and counts the number of times each word in wordArray appears per sentence 
    lineCount = 0
    wordMatrix = np.zeros(shape=(len(currList),11151))
    for i in currList:
        for j in i:
            j = j.lower()
            if(j == 'ham'):
                wordMatrix[lineCount, 0] = 1
            elif(j == 'spam'):
                wordMatrix[lineCount, 0] = 0
            else:
                wordIndex = wordArray.index(j)
                wordMatrix[lineCount, wordIndex] += 1
        lineCount += 1

    return wordMatrix    

def createY_set(currList, wordMatrix):
    #creates an array from the chosen list containing only the labels 0 or 1 which represent 'spam' or 'ham' 
    Y_set = []
    for i in range(0, len(currList)):
        Y_set.append(wordMatrix[i, 0])

    return Y_set

def createx_set(currList, wordArrayNoTarget):
    #similiar to createMatrixWordCound but passes the label values
    lineCount = 0
    lines = len(currList)
    amountWords = len(wordArrayNoTarget)

    x_set = np.zeros(shape=(lines,amountWords))

    for i in currList:
        for j in i:
            j = j.lower()
            if(j == 'ham'):
                pass
            elif(j == 'spam'):
                pass
            else:
                WordIndex = wordArrayNoTarget.index(j)
                x_set[lineCount, WordIndex] += 1
        lineCount += 1

    return x_set

def findBestHyperParam(validationSplit,x_validation,Y_validation):
    #finds the best values for C, the penalty value, and for the amount of iterations
    maxTestIter = 40
    bestScore = 0
    paramCTest = [1E2,1E3,1E4,1E5]

    for i in range(1,maxTestIter):
        for j in range(0,len(paramCTest) - 1):
    
            clf = SVC(gamma = 'auto',C= paramCTest[j] , max_iter = i)
            clf.fit(x_validation,Y_validation)
            tempScore = clf.score(x_validation, Y_validation, sample_weight=None)

            if(tempScore > bestScore):
                bestScore = tempScore
                bestC = paramCTest[j]
                bestIter = i
                print("\nFound new best values with highest precision " + str(bestScore))
                print("Best C: " + str(bestC) + " best iteration: " + str(bestIter))

    return bestC, bestIter          

def lineConvert(currList,wordArray,wordArrayNoTarget):
    #similiar to createx_set except it returns one line at a time
    wordMatrix = np.zeros(shape=(1,len(wordArrayNoTarget)))

    for i in currList:
        i = i.lower()
        if(i == 'ham'):
            pass
        elif(i == 'spam'):
            pass
        else:
            wordIndex = wordArray.index(i)
            wordMatrix[0, wordIndex] += 1
    
    return wordMatrix

def testMethod(currList,clf,wordArray,wordArrayNoTarget):
    #test the precision of the model using the predict function from the sklearn.svm library
    totalGuesses = 0
    correctGuesses = 0
    totalPrecision = 0
    currLabel = 0

    for i in currList:
        if(i[0] == 'ham'):
            currLabel = 1  
        else:
            currLabel = 0
        
        testMatrix = lineConvert(i,wordArray,wordArrayNoTarget)
        prediction = clf.predict(testMatrix)
        totalGuesses +=1
        if(prediction == currLabel):
            correctGuesses +=1

    totalPrecision = (correctGuesses/totalGuesses) * 100

    return totalPrecision

def mainSVC():

    #time execution
    start = time.time()

    #load bag of words
    curatedList = pickle.load( open("curatedCollection", "rb"))
    #create array with all words
    wordArray = createWordArray(curatedList)
    #create array with all words except 'target'
    wordArrayNoTarget = createWordArrayNoTarget(curatedList)

    #make splits
    trainSplit,testSplit,validationSplit = makeSplits(curatedList) 

    #start validation
    print("###### Starting Validation ######\n")
    validationMatrix = createMatrixWordCount(validationSplit,wordArray)
    Y_validation = createY_set(validationSplit,validationMatrix)
    x_validation = createx_set(validationSplit,wordArrayNoTarget)
    bestC, bestIter = findBestHyperParam(validationSplit,x_validation,Y_validation)
    print("###### Validation Finished ######\n")
    print("Best values found\nC: " + str(bestC) + "\nMax number of iterations:" + str(bestIter))

    #start train
    trainMatrix = createMatrixWordCount(trainSplit,wordArray)
    Y_train = createY_set(trainSplit,trainMatrix)
    x_train = createx_set(trainSplit,wordArrayNoTarget)

    #fit the model
    clf = SVC(gamma = 'auto',C= bestC,max_iter= bestIter)
    clf.fit(x_train,Y_train)
    trainScore = clf.score(x_train, Y_train, sample_weight=None)
    print("\nTrain finished, score: " + str(round(trainScore,2)))

    #test model
    totalPrecision = testMethod(testSplit,clf,wordArray,wordArrayNoTarget)
    print("\nTesting finished, total precision for SVC: " + str(totalPrecision) +"%")

    end = time.time()
    final = end - start
    print("\nExecution time of : " + str(final) + " seconds")

if __name__ == "__main__":
   mainSVC()