import pickle
import math
import time
          
def createDict(curList, keyword):
    #iterares through each inner array and creates dictionary based on keyword, 'ham' or 'spam'
    line_count = 0
    tempDict = {}
    for i in curList:
       
        if(i[0] == keyword):
            tempList = i.copy()
            
            line_count += 1
            del tempList[0]
            for j in tempList:
                currWord = j.lower()
                if currWord in tempDict:
                    tempDict[currWord] += 1
                else:
                    tempDict[currWord] = 1
    
    return(tempDict)
    
def probability(num, total):
	probability = num / total
	
	return probability
			
def getDictSize(testDict):
    total = 0
    for i in testDict:
        
        total += testDict[i]
    
    return total

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

    return trainSplit,testSplit,validationSplit

def threshold(hamActualCount,spamActualCount,hyperParamC):

    b = math.log(hyperParamC) + ( math.log(hamActualCount) - math.log(spamActualCount))
   
    return b

def totalWordCount(word,hamDict,spamDict):
    
    totalCountInHam = 0
    totalCountInSpam = 0 
    totalCount = 0
    word = word.lower()

    if word in hamDict:
        totalCountInHam += hamDict[word]

    if word in spamDict:       
        totalCountInSpam += spamDict[word]
 
    totalCount = totalCountInHam + totalCountInSpam

    return totalCount                

def hamOrSpam(hamsize,spamsize,hamDict,spamDict, testString,t):

    del testString[0]
    testString = [x.lower() for x in testString]
    logHam = 0 
    logSpam = 0
    
    for i in testString:
        wordCount= totalWordCount(i,hamDict,spamDict)
        tempProbH = 0
        tempProbS = 0

        if(i in hamDict and i in spamDict):
 
            tempProbH = probability(hamDict[i],hamsize)
            tempProbS = probability(spamDict[i],spamsize)
            logHam = math.log(tempProbH)
            logSpam = math.log(tempProbS)
            
        elif(i not in spamDict and i in hamDict):
    
            tempProbH = probability(hamDict[i],hamsize)
            tempProbS = 0
            logHam = math.log(tempProbH)
            logSpam = 0
            
        elif(i not in hamDict and i in spamDict):   
    
            tempProbH = 0
            tempProbS = probability(spamDict[i],spamsize)
            logHam = 0
            logSpam = math.log(tempProbS)

        t += t + wordCount * ( logSpam - logHam )
    
    return 'ham' if t < 0 else 'spam'

def listRate(hamsize,spamsize,hamDict,spamDict, curatedList,startingC,validationFlag,iterations):
    spamRecvCount = 0
    hamRecvCount = 0
    spamActualCount = 0
    hamActualCount = 0
    successCountHam = 0
    successCountSpam = 0 
    hyperParamC = startingC    
    finalSuccessRate = 0
    maxSuccess = 0 
    bestC = 0
    totalCount = 0
    
    for i in curatedList:
        totalCount += 1
        if (i[0] == 'ham'):
            hamActualCount += 1
        elif(i[0] == 'spam'):
            spamActualCount += 1
        else:
            print('There was mistake!')
            break
        
    for i in range(iterations):
        
        t=  - (threshold(hamActualCount,spamActualCount,hyperParamC))
               
        for i in curatedList:
            
            cpI = i.copy()
            
            result = hamOrSpam(hamsize,spamsize,hamDict,spamDict, cpI,t)
            
            if(result == 'ham'):
                hamRecvCount += 1
            else:
                spamRecvCount += 1
            
            if(i[0] == result):
                if(result == 'ham'):
                    successCountHam += 1
                else:
                    successCountSpam += 1
                    
        finalSuccessRate = ((successCountSpam + successCountHam) / (totalCount)) * 100
        
        if(maxSuccess < finalSuccessRate):
            maxSuccess = finalSuccessRate
            bestC = hyperParamC
            print("\nNew biggest sucess rate of: " + str(maxSuccess) + " using C = " + str(bestC))
        
        if(validationFlag):
            hyperParamC += 1

        hamRecvCount = 0
        spamRecvCount = 0
        successCountHam = 0
        successCountSpam = 0
    
    print("\nFinal max value for sucess rate : " + str(maxSuccess) + " using C = " + str(format(round(bestC,2))))    

    return bestC,maxSuccess

def validationSequence(validationSplit,startingC):

    validationHam = createDict(validationSplit, 'ham')
    validationSpam = createDict(validationSplit, 'spam')

    validationHamsize = getDictSize(validationHam)
    validationSpamsize = getDictSize(validationSpam)

    bestC,success = listRate(validationHamsize,validationSpamsize,validationHam,validationSpam,validationSplit,startingC,True,500)

    return bestC

def mainNaive():

    #time execution
    start = time.time()

    #load bag of words
    curatedList = pickle.load( open("curatedCollection", "rb"))

    #make splits, 70% train 15/15 test and validation
    print("###### Starting Validation ######\n")
    trainSplit,testSplit,validationSplit = makeSplits(curatedList) 
    bestC = validationSequence(validationSplit,0.1)
    print("###### Validation Finished ######\n")

    #train with trainsplit   
    hamDict = createDict(trainSplit, 'ham')
    spamDict = createDict(trainSplit, 'spam')
    
    hamsize = getDictSize(hamDict)
    spamsize = getDictSize(spamDict)
    
    #test naive bayes using test split
    finalC, finalRate = listRate(hamsize,spamsize,hamDict,spamDict,testSplit,bestC,False,1)
    
    end = time.time()
    final = end - start
    print("\nExecution time of : " + str(final) + " seconds")

if __name__ == "__main__":
    mainNaive()