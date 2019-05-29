import pickle
import math
import numpy as np
import time

def setInitialWeights(currList):    
    tempDict ={}
    
    for i in currList:
        ignoreStart = 0
        for j in i:
            currWord = j.lower()
            if ignoreStart != 0:
                if currWord not in tempDict:
                    tempDict[currWord] = 0
            ignoreStart = 1
    
    return tempDict

def absoluteFreq(currList):
    tempDict ={}
    
    for i in currList:
        ignoreStart = 0
        for j in i:
            currWord = j.lower()
            if ignoreStart != 0:
                if currWord not in tempDict:
                    tempDict[currWord] = 1
                else:
                    tempDict[currWord] += 1
            ignoreStart = 1
            
    return tempDict

def makeSplits(currList):
    trainSplit = 0
    testSplit = 0
    validationSplit = 0

    totalLines = len(currList)
    firstSplit = round(totalLines*0.7)

    trainSplit = currList[0:firstSplit]

    secondSplit = firstSplit + round(totalLines*0.15)

    testSplit = currList[firstSplit : secondSplit]

    validationSplit = currList[secondSplit : totalLines]

    #print(len(trainSplit))
    #print(len(testSplit))
    #print(len(validationSplit))

    return trainSplit,testSplit,validationSplit

def perceptronMakeWeights(currList,weightDict,freqDict,maxIter,bias):
    
    currLabel = 0
  
    for g in range(maxIter):
        for i in currList:
            a = 0
            tempList = i.copy()
            stringLabel = tempList[0]
            if(stringLabel == 'ham'): ##currLabel corresponde a 1 se for ham ou a -1 se for spam 
                currLabel = 1
            else:
                currLabel = -1 
            del tempList[0]
            
            for j in tempList:
                currWord = j.lower()
                a = a + (( weightDict[currWord] * freqDict[currWord]) + bias)
            
            if(currLabel*a <= 0 ):
                bias = bias + currLabel
                #print("Bias = " + str(bias))
                #print("currLabel = " + str(currLabel))
                for j in tempList:
                    currWord = j.lower()
                    weightDict[currWord] = weightDict[currWord] + (currLabel * freqDict[currWord])
                    #print("For " + currWord + " = " + str(weightDict[currWord]))
                
    return weightDict, bias  
          
def perceptronTest(weights,freqDict,bias,testList):
    a = 0
    totalGuesses = 0
    correctGuesses = 0
    totalPrecision = 0

    for i in testList:
        a = 0
        tempList = i.copy()
        stringLabel = tempList[0]
        
        if(stringLabel == 'ham'): ##currLabel corresponde a 1 se for ham ou a -1 se for spam 
            currLabel = 1
        else:
            currLabel = -1 
        del tempList[0]

        for j in tempList:
            currWord = j.lower()     
            a = a + (( weights[currWord] * freqDict[currWord]) + bias)
            totalGuesses += 1

            if(np.sign(a) == currLabel):
                correctGuesses += 1

    totalPrecision = (correctGuesses/totalGuesses) * 100

    return totalPrecision             

def perceptronValidation(currList,weightDict,freqDict,startingBias,rangeVal,maxIter):
    minBias = startingBias - rangeVal
    maxBias = startingBias + rangeVal
    maxPrecision = 0
    bestBias = 0
    bestIter = 0
   
    for currentBias in range(minBias,maxBias+1):
        tempWeights = weightDict
        bias = currentBias
        
        for currentIter in range(maxIter):
            for i in currList:
                a = 0
                tempList = i.copy()
                stringLabel = tempList[0]
                if(stringLabel == 'ham'): ##currLabel corresponde a 1 se for ham ou a -1 se for spam 
                    currLabel = 1
                else:
                    currLabel = -1 
                del tempList[0]
                
                for j in tempList:
                    currWord = j.lower()
                    a = a + (( tempWeights[currWord] * freqDict[currWord]) + bias)
                
                if(currLabel*a <= 0 ):
                    bias = bias + currLabel

                    for j in tempList:
                        currWord = j.lower()
                        tempWeights[currWord] = tempWeights[currWord] + (currLabel * freqDict[currWord])
            
            precision = perceptronTest(tempWeights,freqDict,bias,currList)   

            if(maxPrecision < precision):
                print("\nFound new best values with highest precision " + str(precision))
                print("Best bias: " + str(currentBias) + " best iteration: " + str(currentIter))
                maxPrecision = precision
                bestBias = currentBias
                bestIter = currentIter

    return bestIter, bestBias

def mainPer():
    start = time.time()
    #initial large list containing smaller lists with sentences
    curatedList = pickle.load( open("curatedCollection", "rb"))
    #define splits
    trainSplit,testSplit,validationSplit = makeSplits(curatedList)
    #create dict with all words with weights = 0
    initialWeights = setInitialWeights(curatedList)
    #calc freq absolute
    freqDict = absoluteFreq(curatedList)
    #validation 
    print("###### Starting Validation ######\n")
    bestMaxIterator, bestBias = perceptronValidation(validationSplit,initialWeights,freqDict,-10,25,10)
    print("Best value for total iterations : " + str(bestMaxIterator)) 
    print("Best value for starting point of bias : " + str(bestBias))
    print("###### Validation Finished ######\n")
    #perceptron train
    finalWeights, finalBias = perceptronMakeWeights(trainSplit,initialWeights,freqDict,bestMaxIterator,bestBias) 
    #test method
    testPrecision = perceptronTest(finalWeights,freqDict,finalBias,testSplit)
    print("Final precision of perceptron : " + str(testPrecision))
    end = time.time()
    final = end - start
    print("Execution time of : " + str(final) + " seconds")
    
if __name__ == "__main__":
   mainPer()   