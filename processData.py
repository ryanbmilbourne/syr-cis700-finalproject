#!/usr/bin/python

'''
    @author Tanner Leach
    @date   September 4, 2017
    @brief  Takes in plain text data of format "ham/spam <text_message>" and creates matrix.
            Process number of ham/spams, length of messages (mean/std dev), cleans up white spaces,
            and outputs a NumPy Matrix "ham/spam <text_message>" for easier reading in.
'''

# Package delivery
import argparse
import os.path
import numpy as np
import pandas as panduh

#####################
# processArgs
#####################
#Simple proccessor of command line arguments
def processArgs():
    parser = argparse.ArgumentParser(description='Generate production config for an SMTS chassis')
    parser.add_argument('--text', metavar='<spamMessages>.txt', nargs='+',required=True,
                        help='text file that contains format of \'<ham/spam><space><message>\\n\'')
    parser.add_argument('--output', metavar='<file_name>.np', nargs='+',required=False,
                        help='output NumPy Matrix file name')

    args = parser.parse_args()

    return args

#####################
# processData
#####################
def processData(textFile):

    inputData = open(textFile, 'r')
    lines = inputData.readlines()

    #initialize the matrix
    dataMatrix = panduh.DataFrame(columns=['label', 'text'])

    line_count          = 0
    message_len_total   = 0
    ham_count           = 0  
    ham_length          = 0
    spam_count          = 0
    spam_length         = 0
    
    #loop through the lines
    for line in lines:


        #trim any white space at the end or beginning
        line = line.strip()
        #Data set has first word as spam/ham classifier
        splitLine = line.split()

        #should be 'spam' or 'ham'
        classifier = splitLine.pop(0)
        message = ' '.join(splitLine)

        temp_row = panduh
        #make into a 1x2 matrix
        dataMatrix.loc[line_count] = [classifier, message]

        line_count += 1

        #Store in our matrix
        #dataMatrix.append(temp_row)

        #the message is the second index through the last index, but must join indices with ' '

        if "spam" in classifier:
            spam_count  += 1
            spam_length += len(splitLine)
        elif "ham" in classifier:
            ham_count  += 1
            ham_length += len(splitLine)
        else:
            print "ERROR: didnt find ham or spam. I dont want any else to eat."
            exit(-1)

    
        message_len_total += len(splitLine)

    #clean up the initial row of the data matrix that was just a place holder
    #dataMatrix = dataMatrix.drop(0)

    #print "MATRIX: ", dataMatrix.shape
    #print "MATRIX: ", dataMatrix.shape

    print "---------------------------"
    print "Total Number of messages  = ", line_count
    print 'Average Length of message: --', (1.0 * message_len_total)/len(dataMatrix), '--'
    print "---------------------------"
    print "Total Number of ham       = ", ham_count, " (", ((1.0 * ham_count)/len(dataMatrix))* 100, "%)"
    print 'Average Length of ham:     --', (1.0 * ham_length)/ham_count, '--'
    print "---------------------------"
    print "Total Number of spam      = ", spam_count, " (", ((1.0 * spam_count)/len(dataMatrix)) * 100, "%)"
    print 'Average Length of spam:    --', (1.0 * spam_length)/spam_count, '--'
        

    '''
    #add column headers
    headers = ['label','text']
    newData = np.vstack([headers, dataMatrix])

    ### WE LOVE PANDAS ###
    dataFrame = panduh.DataFrame(data=newData[1:,1:],
                                index=newData[1:,0],
                                columns=newData[0,0:])
    '''

    print dataMatrix

    return dataMatrix

def readSentiment(filePath):
    #initialize the matrix
    dataMatrix = panduh.DataFrame(columns=['label', 'text'])

    labels = ['pos','neg']
    loc = 0
    for label in labels:
        inputData = open(filePath+'.'+label)
        lines = inputData.readlines()
        for line in lines:
            line=line.decode('utf-8','ignore').encode("utf-8")
            dataMatrix.loc[loc] = [label, line]
            loc = loc + 1

    dataMatrix = dataMatrix.reindex(np.random.permutation(dataMatrix.index))
    print dataMatrix
    return dataMatrix



#####################
# Main
#####################
if __name__ == "__main__":

    #get cmd line args
    args = processArgs()

    textFile = args.text[0]
    
    if args.output:
        outputFile = args.output[0]
    else:
        outputFile = "none"

    newData = processData(textFile)

    if "none" in outputFile.lower():
        outputFile = textFile.split('.')[0]

    np.save(outputFile, newData)


