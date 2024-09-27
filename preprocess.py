import pandas as pd
import torch

def reduceData(df):
    return df[['ImageID','Respondent Name','AOI Label','Answers']]

def createVocabs(df):
    vocIn = df['AOI Label'].tolist()
    usrs = df['Respondent Name'].tolist()
    targets = df['Answers'].tolist()
    vocAll = []
    vocab = {}
    usrVoc = {}
    targetVoc = {}
    j = 0
    for user in set(usrs):
        usrVoc[user] = j
        j+=1
    i = 1
    for seq in vocIn:
        for tok in seq.lower().split(','):
            vocAll.append(tok.strip())
    for token in set(vocAll):
        vocab[token] = i
        i += 1
    k = 0
    for tgt in set(targets):
        targetVoc[tgt] = k
        k += 1
    return vocab, usrVoc, targetVoc

def applyVocab(df,vocab,usrVocab,targetVoc):
    seqs = df['AOI Label'].tolist()
    users = df['Respondent Name'].map(usrVocab).tolist()
    targets = df['Answers'].map(targetVoc).tolist()
    newSeqs = []
    sizeList = []
    for seq in seqs:
        tmp = []
        for token in seq.lower().split(','):
            tmp.append(vocab[token.strip()])
        newSeqs.append(torch.LongTensor(tmp))
        sizeList.append(len(tmp))
    targets = torch.LongTensor(targets)
    users = torch.LongTensor(users)
    nestedSeqs = torch.nested.nested_tensor(newSeqs)
    return nestedSeqs, targets, users, sizeList

def fullVocabs():
    test = pd.read_csv('rawData/fixations_test.csv',sep='\t')
    train = pd.read_csv('rawData/fixations_train.csv',sep='\t')
    dev = pd.read_csv('rawData/fixations_dev.csv',sep='\t')
    allData = pd.concat([train,test,dev])
    reduced = reduceData(allData)
    vocab, usrVoc, targetVoc = createVocabs(reduced)
    return vocab,usrVoc, targetVoc

if __name__ == '__main__':
    print (torch.cuda.is_available())
    print("Pytorch CUDA Version is ", torch.version.cuda)
    test = pd.read_csv('rawData/fixations_test.csv',sep='\t')
    train = pd.read_csv('rawData/fixations_train.csv',sep='\t')
    dev = pd.read_csv('rawData/fixations_dev.csv',sep='\t')
    vocab, usrVoc, targetVoc = fullVocabs()
    
    trainSeq, trainTgts, trainUsers, trainSizes = applyVocab(train,vocab,usrVoc,targetVoc)
    testSeq, testTgts, testUsers, testSizes = applyVocab(test,vocab,usrVoc,targetVoc)
    devSeq, devTgts, devUsers, devSizes = applyVocab(dev,vocab,usrVoc,targetVoc)

    torch.save(torch.nested.to_padded_tensor(trainSeq,0),'data/latest_train.pt')
    torch.save(trainTgts,'data/latest_trainTgt.pt')
    torch.save(trainUsers,'data/latest_trainUsr.pt')

    torch.save(torch.nested.to_padded_tensor(testSeq,0),'data/latest_test.pt')
    torch.save(testTgts,'data/latest_testTgt.pt')
    torch.save(testUsers,'data/latest_testUsr.pt')

    torch.save(torch.nested.to_padded_tensor(devSeq,0),'data/latest_dev.pt')
    torch.save(devTgts,'data/latest_devTgt.pt')
    torch.save(devUsers,'data/latest_devUsr.pt')

    torch.save(trainSizes,'data/latest_trainSizes.pt')
    torch.save(testSizes,'data/latest_testSizes.pt')
    torch.save(devSizes,'data/latest_devSizes.pt')