import torch
import json

import random as rd
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import BertTokenizer


class IndividualSequenceData(Dataset):
    def __init__(self,subset):
        self.subset = subset
        self.baseDataPath = "visual_genome_image_features/eyetrack/"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        with open("{}/objects_aois.json".format(self.baseDataPath)) as objectDict:
            self.objects = json.load(objectDict)
        with open("{}/phrases.json".format(self.baseDataPath)) as phraseDict:
            self.phrases = json.load(phraseDict)
        with open("{}/userImages_{}.json".format(self.baseDataPath,self.subset)) as phraseDict:
            self.usrImgTgt = json.load(phraseDict)

    def __len__(self):
        return len(self.usrImgTgt.keys())
    
    def __getitem__(self,index):
        img,usr,tgt = self.usrImgTgt[str(index)]
        phrases = self.phrases[str(img)]
        tokenized = self.tokenizer(phrases)
        objects = self.objects[str(img)]
        visualObjects = torch.load("{}{}.pt".format(self.baseDataPath,str(img)))
        visualObjects = self.__removeRemoved__(visualObjects,objects)
        return tokenized,visualObjects,usr,tgt

    def __removeRemoved__(self,visualObjects,objects):
        badIndices = set([i for i, e in enumerate(objects) if e == "removed"])
        indexRange = visualObjects.shape[0]
        goodIndices = set(range(0,indexRange)) - badIndices
        actualImages = torch.index_select(visualObjects,0,torch.tensor(list(goodIndices)))
        return actualImages

def collate_fn_is(batch):
    #0 = phrase, 1 = img features, 2 = user, 3 = target
    longestPhrase = 0
    longestImgSeq = 0
    batchsize = len(batch)
    #checking how much padding is needed; is there a way without iteration?
    for sample in batch:
        if len(sample[0]['input_ids']) > longestPhrase:
            longestPhrase = len(sample[0]['input_ids'])
        if sample[1].shape[0] > longestImgSeq:
            longestImgSeq = sample[1].shape[0]
    #init tensors for batch
    #bert input
    input_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
    type_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
    #bert -> 0 for padding, 1 for non-padding
    att_pad_mask = torch.zeros((batchsize,longestPhrase),dtype=torch.bool)
    #caption target
    captiontargets = torch.empty((batchsize),dtype=torch.long)
    #users
    users = torch.empty((batchsize,1),dtype=torch.long)
    #visual input
    images = torch.zeros((batchsize,longestImgSeq,2048),dtype=torch.float)
    #transformer encoder -> 1 for padding, 0 for non-padding
    img_pad_mask = torch.ones((batchsize,longestImgSeq),dtype=torch.bool)
    i = 0
    for sample in batch:
        #image sequence
        currentImgLen = sample[1].shape[0]
        images[i][:currentImgLen] = sample[1]
        img_pad_mask[i][:currentImgLen] = torch.zeros(currentImgLen)
        #text sequence
        currentTextLen = len(sample[0]['input_ids'])
        #print (sample[1])
        input_ids[i][:currentTextLen] = torch.IntTensor(sample[0]['input_ids'])
        att_pad_mask[i][:currentTextLen] = torch.ones(currentTextLen)
        #targets
        captiontargets[i] = torch.LongTensor([sample[3]])
        #users
        users[i] = torch.LongTensor([sample[2]])
        i += 1
    text = {'input_ids' : input_ids, 'attention_mask' : att_pad_mask, 'token_type_ids' : type_ids}
    #reverse mask for torch transformer
    rev_att_pad_mask = ~att_pad_mask
    return images,img_pad_mask,text,rev_att_pad_mask,captiontargets,users

class EnsembleData(Dataset):
    def __init__(self,subset):
        ###FEATURE PART
        self.subset = subset
        self.baseDataPath = "visual_genome_image_features/eyetrack/"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        with open("{}/objects_aois.json".format(self.baseDataPath)) as objectDict:
            self.objects = json.load(objectDict)
        with open("{}/phrases.json".format(self.baseDataPath)) as phraseDict:
            self.phrases = json.load(phraseDict)
        with open("{}/userImages_{}.json".format(self.baseDataPath,self.subset)) as phraseDict:
            self.usrImgTgt = json.load(phraseDict)
        ###SEQUENCE PART
        self.sequenceData = torch.load('{}latest_{}.pt'.format(self.baseDataPath,subset))
        self.sequenceLengths = torch.load('{}latest_{}Sizes.pt'.format(self.baseDataPath,subset))
        self.sequenceTargets = torch.load('{}latest_{}Tgt.pt'.format(self.baseDataPath,subset))
        self.sequenceUser = torch.load('{}latest_{}Usr.pt'.format(self.baseDataPath,subset))

    def __len__(self):
        return len(self.usrImgTgt.keys())
    
    def __getitem__(self,index):
        ###FEATURE PART
        img,usr,tgt = self.usrImgTgt[str(index)]
        phrases = self.phrases[str(img)]
        tokenized = self.tokenizer(phrases)
        objects = self.objects[str(img)]
        visualObjects = torch.load("{}{}.pt".format(self.baseDataPath,str(img)))
        visualObjects = self.__removeRemoved__(visualObjects,objects)
        ###SEQUENCE PART
        inputs = self.sequenceData[index]
        targets = self.sequenceTargets[index]
        lens = self.sequenceLengths[index]
        user = self.sequenceUser[index]
        return tokenized,visualObjects,usr,tgt,inputs,targets,lens,user

    #For Feature Part
    def __removeRemoved__(self,visualObjects,objects):
        badIndices = set([i for i, e in enumerate(objects) if e == "removed"])
        indexRange = visualObjects.shape[0]
        goodIndices = set(range(0,indexRange)) - badIndices
        actualImages = torch.index_select(visualObjects,0,torch.tensor(list(goodIndices)))
        return actualImages
    
def collate_ensemble(batch):
    #[0][4] = sequence / [0][5] = target / [0][6] = sequence length / [0][7] user 
    seqMax = batch[0][4].shape[0]
    batchData = torch.zeros((len(batch),seqMax),dtype=torch.long)
    targets = torch.zeros(len(batch),dtype=torch.long)
    users = torch.zeros((len(batch),1),dtype=torch.long)
    lengths = []
    i = 0
    longestPhrase = 0
    longestImgSeq = 0
    batchsize = len(batch)
    for sample in batch:
        batchData[i] = sample[4]
        targets[i] = sample[5]
        users[i] = sample[7]
        lengths.append(sample[6])
        i += 1
        if len(sample[0]['input_ids']) > longestPhrase:
            longestPhrase = len(sample[0]['input_ids'])
        if sample[1].shape[0] > longestImgSeq:
            longestImgSeq = sample[1].shape[0]
    input_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
    type_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
    att_pad_mask = torch.zeros((batchsize,longestPhrase),dtype=torch.bool)
    captiontargets = torch.empty((batchsize),dtype=torch.long)
    users = torch.empty((batchsize,1),dtype=torch.long)
    images = torch.zeros((batchsize,longestImgSeq,2048),dtype=torch.float)
    img_pad_mask = torch.ones((batchsize,longestImgSeq),dtype=torch.bool)
    i = 0
    for sample in batch:
        currentImgLen = sample[1].shape[0]
        images[i][:currentImgLen] = sample[1]
        img_pad_mask[i][:currentImgLen] = torch.zeros(currentImgLen)
        currentTextLen = len(sample[0]['input_ids'])
        input_ids[i][:currentTextLen] = torch.IntTensor(sample[0]['input_ids'])
        att_pad_mask[i][:currentTextLen] = torch.ones(currentTextLen)
        captiontargets[i] = torch.LongTensor([sample[3]])
        users[i] = torch.LongTensor([sample[2]])
        i += 1
    text = {'input_ids' : input_ids, 'attention_mask' : att_pad_mask, 'token_type_ids' : type_ids}
    rev_att_pad_mask = ~att_pad_mask
    seqIns = (batchData.transpose(0,1), targets, lengths, users.transpose(0,1))
    featureIns = (images,img_pad_mask,text,rev_att_pad_mask,captiontargets,users)
    return seqIns,featureIns

class EyeTrackFullData(Dataset):
    def __init__(self,subset):
        self.subset = subset
        self.baseDataPath = "visual_genome_image_features/eyetrack/"
        self.sequenceData = torch.load('{}latest_{}.pt'.format(self.baseDataPath,subset))
        self.sequenceLengths = torch.load('{}latest_{}Sizes.pt'.format(self.baseDataPath,subset))
        self.sequenceTargets = torch.load('{}latest_{}Tgt.pt'.format(self.baseDataPath,subset))
        self.sequenceUser = torch.load('{}latest_{}Usr.pt'.format(self.baseDataPath,subset))
    
    def __len__(self):
        return (len(self.sequenceLengths))
    
    def __getitem__(self,index):
        inputs = self.sequenceData[index]
        targets = self.sequenceTargets[index]
        lens = self.sequenceLengths[index]
        user = self.sequenceUser[index]
        return inputs,targets,lens,user

def collate_full_seq(batch):
    #[0][0] = sequence / [0][1] = target / [0][2] = sequence length / [0][3] user 
    seqMax = batch[0][0].shape[0]
    batchData = torch.zeros((len(batch),seqMax),dtype=torch.long)
    targets = torch.zeros(len(batch),dtype=torch.long)
    users = torch.zeros((len(batch),1),dtype=torch.long)
    lengths = []
    i = 0
    for sample in batch:
        batchData[i] = sample[0]
        targets[i] = sample[1]
        users[i] = sample[3]
        lengths.append(sample[2])
        i += 1
    return batchData.transpose(0,1), targets, lengths, users.transpose(0,1)

class IndividualTransitionData(Dataset):
    def __init__(self,subset):
        self.subset = subset
        self.baseDataPath = "visual_genome_image_features/eyetrack/"
        self.baseTMPath = "visual_genome_image_features/individual_transitions/"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        with open("{}/objects_aois.json".format(self.baseDataPath)) as objectDict:
            self.objects = json.load(objectDict)
        with open("{}/phrases.json".format(self.baseDataPath)) as phraseDict:
            self.phrases = json.load(phraseDict)
        with open("{}/userImages_{}.json".format(self.baseDataPath,self.subset)) as sampleDict:
            self.usrImgTgt = json.load(sampleDict)
        with open("{}/userImages_invUserDict.json".format(self.baseDataPath)) as userDict:
            self.usrToNmbr = json.load(userDict)

    def __len__(self):
        return len(self.usrImgTgt.keys())
    
    def __getitem__(self,index):
        img,usr,tgt = self.usrImgTgt[str(index)]
        userId = self.usrToNmbr[str(usr)]
        phrases = self.phrases[str(img)]
        tokenized = self.tokenizer(phrases)
        objects = self.objects[str(img)]
        visualObjects = torch.load("{}{}.pt".format(self.baseDataPath,str(img)))
        visualObjects = self.__removeRemoved__(visualObjects,objects)
        transitions = torch.load("{}{}_{}.pt".format(self.baseTMPath,img,userId))
        return tokenized,visualObjects,tgt,transitions,usr
    
    def __removeRemoved__(self,visualObjects,objects):
        badIndices = set([i for i, e in enumerate(objects) if e == "removed"])
        indexRange = visualObjects.shape[0]
        goodIndices = set(range(0,indexRange)) - badIndices
        actualImages = torch.index_select(visualObjects,0,torch.tensor(list(goodIndices)))
        return actualImages
    

class ITDCollator(object):
    def __init__(self, *params):
        self.params = params
    def __call__(self, batch):
        #0 = phrase, 1 = img features, 2 = target, 3 = transition Matrix
        longestPhrase = 0
        longestImgSeq = 0
        batchsize = len(batch)
        for sample in batch:
            if len(sample[0]['input_ids']) > longestPhrase:
                longestPhrase = len(sample[0]['input_ids'])
            if sample[1].shape[0] > longestImgSeq:
                longestImgSeq = sample[1].shape[0]
        longestMatrix = longestPhrase + longestImgSeq
        #init tensors for batch
        #transition matrix
        transitionMatrix = torch.zeros((batchsize,longestMatrix,longestMatrix),dtype=torch.float)
        input_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
        type_ids = torch.zeros((batchsize,longestPhrase),dtype=torch.int)
        att_pad_mask = torch.zeros((batchsize,longestPhrase),dtype=torch.bool)
        captiontargets = torch.empty((batchsize),dtype=torch.long)
        images = torch.zeros((batchsize,longestImgSeq,2048),dtype=torch.float)
        img_pad_mask = torch.ones((batchsize,longestImgSeq),dtype=torch.bool)
        #users
        users = torch.empty((batchsize,1),dtype=torch.long)
        i = 0
        for sample in batch:
            #image sequence
            currentImgLen = sample[1].shape[0]
            images[i][:currentImgLen] = sample[1]
            img_pad_mask[i][:currentImgLen] = torch.zeros(currentImgLen)
            #text sequence
            currentTextLen = len(sample[0]['input_ids'])
            input_ids[i][:currentTextLen] = torch.IntTensor(sample[0]['input_ids'])
            att_pad_mask[i][:currentTextLen] = torch.ones(currentTextLen)
            #targets
            captiontargets[i] = sample[2]
            #users
            users[i] = torch.LongTensor([sample[4]])
            #matrix
            txtOnly = sample[3][:currentTextLen,:currentTextLen]
            imgOnly = sample[3][-currentImgLen:,-currentImgLen:]
            lowerLeft = sample[3][-currentImgLen:,:currentTextLen]
            upperRight = sample[3][:currentTextLen,-currentImgLen:]
            transitionMatrix[i][:currentTextLen,:currentTextLen] = txtOnly
            transitionMatrix[i][longestPhrase:longestPhrase+currentImgLen,longestPhrase:longestPhrase+currentImgLen] = imgOnly
            transitionMatrix[i][longestPhrase:longestPhrase+currentImgLen,:currentTextLen] = lowerLeft
            transitionMatrix[i][:currentTextLen,longestPhrase:longestPhrase+currentImgLen] = upperRight

            i += 1
        text = {'input_ids' : input_ids, 'attention_mask' : att_pad_mask, 'token_type_ids' : type_ids}
        rev_att_pad_mask = ~att_pad_mask
        ###MIRRORED
        transitionMatrix = transitionMatrix*self.params[0]
        transitionMatrix = transitionMatrix.add(transitionMatrix.transpose(1,2))
        return images,img_pad_mask,text,rev_att_pad_mask,captiontargets,transitionMatrix, users