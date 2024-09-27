import torch
import loader as nl
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

class IndividualSequenceTrainer():
    def __init__(self,batchSize,nrEpochs,learningRate):
        self.trainData = "train"
        self.validationData = "dev"
        self.testData = "test"
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.evalAll = 1

    def train(self,model):
        print ("~~~TRAINING~~~")
        device = model.device
        trainingData = nl.IndividualSequenceData(self.trainData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_fn_is,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),self.learningRate)
        model.train()
        bestModel = None
        bestValLoss = 100
        currValLoss = 100
        bestEpoch = 0
        for x in range(self.nrEpochs):
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)

                loss = lossFunction(outs,targets)
                
                loss.backward()
                optimizer.step()

                lossAbs += float(loss)
                nrBatches += 1
                optimizer.zero_grad()
            print ("EPOCH: {}".format(x+1))
            print ("Loss: {}".format(lossAbs/nrBatches))
            if x % self.evalAll == 0:
                currValLoss = self.validate(model)
                if currValLoss < bestValLoss:
                    bestModel = model
                    bestEpoch = x + 1
                    bestValLoss = currValLoss
        if self.evalAll != 1:
            currValLoss = self.validate(model)
            if currValLoss < bestValLoss:
                bestModel = model
                bestEpoch = self.nrEpochs+1
                bestValLoss = currValLoss
        print ("Best Model in Epoch {} @ {}".format(bestEpoch,bestValLoss))
        return bestModel
    
    def validate(self,model):
        device = model.device
        trainingData = nl.IndividualSequenceData(self.validationData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_fn_is,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        with torch.no_grad():
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)

                loss = lossFunction(outs,targets)
                
                lossAbs += float(loss)
                nrBatches += 1
            print ("\tEval Loss: {}".format(lossAbs/nrBatches))
        return lossAbs/nrBatches
    
    def test(self,model):
        device = model.device
        trainingData = nl.IndividualSequenceData(self.testData)
        loader = DataLoader(trainingData,613,num_workers=48,shuffle=True,collate_fn=nl.collate_fn_is,pin_memory=True)
        model.eval()
        with torch.no_grad():
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)
                predictions = torch.topk(outs,1).indices.view(-1)
                ###Eval classification
                print ("Test ACC: {}".format((predictions.shape[0]-torch.count_nonzero(torch.sub(predictions,targets)))/predictions.shape[0]))

class EnsembleTrainer():
    def __init__(self,batchSize,nrEpochs,learningRate):
        self.trainData = "train"
        self.validationData = "dev"
        self.testData = "test"
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.evalAll = 1

    def train(self,model):
        print ("~~~TRAINING~~~")
        device = model.device
        trainingData = nl.EnsembleData(self.trainData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_ensemble,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),self.learningRate)
        model.train()
        bestModel = None
        bestValLoss = 100
        currValLoss = 100
        bestEpoch = 0
        for x in range(self.nrEpochs):
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                #Features
                sample[1][1][2]['input_ids'] = sample[1][1][2]['input_ids'].to(device)
                sample[1][1][2]['token_type_ids'] = sample[1][1][2]['token_type_ids'].to(device)
                sample[1][1][2]['attention_mask'] = sample[1][1][2]['attention_mask'].to(device)
                featureInputs = sample[1][1][0].to(device),sample[1][1][1].to(device),sample[1][1][2],sample[1][1][3].to(device),sample[1][1][5].to(device)
                #Sequences
                ins = sample[1][0][0].to(device)
                lengths = sample[1][0][2]
                users = sample[1][0][3].to(device)
                seqInputs = ins,lengths,users
                #Feed to Model
                outs = model(seqInputs,featureInputs)
                targets = sample[1][0][1].to(device)

                loss = lossFunction(outs,targets)
                
                loss.backward()
                optimizer.step()

                lossAbs += float(loss)
                nrBatches += 1
                optimizer.zero_grad()
            print ("EPOCH: {}".format(x+1))
            print ("Loss: {}".format(lossAbs/nrBatches))
            if x % self.evalAll == 0:
                currValLoss = self.validate(model)
                if currValLoss < bestValLoss:
                    bestModel = model
                    bestEpoch = x + 1
                    bestValLoss = currValLoss
        if self.evalAll != 1:
            currValLoss = self.validate(model)
            if currValLoss < bestValLoss:
                bestModel = model
                bestEpoch = self.nrEpochs+1
                bestValLoss = currValLoss
        print ("Best Model in Epoch {} @ {}".format(bestEpoch,bestValLoss))
        return bestModel
    
    def validate(self,model):
        device = model.device
        trainingData = nl.EnsembleData(self.validationData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_ensemble,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        with torch.no_grad():
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                sample[1][1][2]['input_ids'] = sample[1][1][2]['input_ids'].to(device)
                sample[1][1][2]['token_type_ids'] = sample[1][1][2]['token_type_ids'].to(device)
                sample[1][1][2]['attention_mask'] = sample[1][1][2]['attention_mask'].to(device)
                featureInputs = sample[1][1][0].to(device),sample[1][1][1].to(device),sample[1][1][2],sample[1][1][3].to(device),sample[1][1][5].to(device)
                #Sequences
                ins = sample[1][0][0].to(device)
                lengths = sample[1][0][2]
                users = sample[1][0][3].to(device)
                seqInputs = (ins,lengths,users)
                #Feed to Model
                outs = model(seqInputs,featureInputs)
                targets = sample[1][0][1].to(device)

                loss = lossFunction(outs,targets)
                
                lossAbs += float(loss)
                nrBatches += 1
            print ("\tEval Loss: {}".format(lossAbs/nrBatches))
        return lossAbs/nrBatches
    
    def test(self,model):
        device = model.device
        trainingData = nl.EnsembleData(self.testData)
        loader = DataLoader(trainingData,613,num_workers=48,shuffle=True,collate_fn=nl.collate_ensemble,pin_memory=True)
        model.eval()
        with torch.no_grad():
            for sample in enumerate(loader):
                sample[1][1][2]['input_ids'] = sample[1][1][2]['input_ids'].to(device)
                sample[1][1][2]['token_type_ids'] = sample[1][1][2]['token_type_ids'].to(device)
                sample[1][1][2]['attention_mask'] = sample[1][1][2]['attention_mask'].to(device)
                featureInputs = sample[1][1][0].to(device),sample[1][1][1].to(device),sample[1][1][2],sample[1][1][3].to(device),sample[1][1][5].to(device)
                #Sequences
                ins = sample[1][0][0].to(device)
                lengths = sample[1][0][2]
                users = sample[1][0][3].to(device)
                seqInputs = (ins,lengths,users)
                #Feed to Model
                outs = model(seqInputs,featureInputs)
                targets = sample[1][0][1].to(device)

                predictions = torch.topk(outs,1).indices.view(-1)
                ###Eval classification
                print ("Test ACC: {}".format((predictions.shape[0]-torch.count_nonzero(torch.sub(predictions,targets)))/predictions.shape[0]))

class IndividualSequenceTrainerBias():
    def __init__(self,batchSize,nrEpochs,learningRate,ampFactor):
        self.trainData = "train"
        self.validationData = "dev"
        self.testData = "test"
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.evalAll = 1
        self.collator = nl.ITDCollator(ampFactor)

    def train(self,model):
        print ("~~~TRAINING~~~")
        device = model.device
        trainingData = nl.IndividualTransitionData(self.trainData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=self.collator,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),self.learningRate)
        model.train()
        bestModel = None
        bestValLoss = 100
        currValLoss = 100
        bestEpoch = 0
        for x in range(self.nrEpochs):
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device),sample[1][6].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)
                loss = lossFunction(outs,targets)
                
                loss.backward()
                optimizer.step()

                lossAbs += float(loss)
                nrBatches += 1
                optimizer.zero_grad()
            print ("EPOCH: {}".format(x+1))
            print ("Loss: {}".format(lossAbs/nrBatches))
            if x % self.evalAll == 0:
                currValLoss = self.validate(model)
                if currValLoss < bestValLoss:
                    bestModel = model
                    bestEpoch = x + 1
                    bestValLoss = currValLoss
        if self.evalAll != 1:
            currValLoss = self.validate(model)
            if currValLoss < bestValLoss:
                bestModel = model
                bestEpoch = self.nrEpochs+1
                bestValLoss = currValLoss
        print ("Best Model in Epoch {} @ {}".format(bestEpoch,bestValLoss))
        return bestModel
    
    def validate(self,model):
        device = model.device
        trainingData = nl.IndividualTransitionData(self.validationData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=self.collator,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        with torch.no_grad():
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device),sample[1][6].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)

                loss = lossFunction(outs,targets)
                
                lossAbs += float(loss)
                nrBatches += 1
            print ("\tEval Loss: {}".format(lossAbs/nrBatches))
        return lossAbs/nrBatches
    
    def test(self,model):
        device = model.device
        trainingData = nl.IndividualTransitionData(self.testData)
        loader = DataLoader(trainingData,613,num_workers=48,shuffle=True,collate_fn=self.collator,pin_memory=True)
        model.eval()
        with torch.no_grad():
            for sample in enumerate(loader):
                sample[1][2]['input_ids'] = sample[1][2]['input_ids'].to(device)
                sample[1][2]['token_type_ids'] = sample[1][2]['token_type_ids'].to(device)
                sample[1][2]['attention_mask'] = sample[1][2]['attention_mask'].to(device)
                inputs = sample[1][0].to(device),sample[1][1].to(device),sample[1][2],sample[1][3].to(device),sample[1][5].to(device),sample[1][6].to(device)
                outs = model(inputs)
                targets = sample[1][4].to(device)
                predictions = torch.topk(outs,1).indices.view(-1)
                ###Eval classification
                print ("Test ACC: {}".format((predictions.shape[0]-torch.count_nonzero(torch.sub(predictions,targets)))/predictions.shape[0]))

class LSTMTrainer():
    def __init__(self,batchSize,nrEpochs,learningRate):
        self.trainData = "train"
        self.validationData = "dev"
        self.testData = "test"
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.evalAll = 1

    def train(self,model):
        print ("~~~TRAINING~~~")
        device = model.device
        trainingData = nl.EyeTrackFullData(self.trainData)
        loader = DataLoader(trainingData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_full_seq,pin_memory=True)
        #lossFunction = nn.CrossEntropyLoss(weight=torch.Tensor([1,3.3,5.2]).to(device))
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),self.learningRate)
        model.train()
        bestModel = None
        bestValLoss = 100
        currValLoss = 100
        bestEpoch = 0
        for x in range(self.nrEpochs):
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                ins = sample[1][0].to(device)
                lengths = sample[1][2]
                users = sample[1][3].to(device)

                outs = model(ins,lengths,users)
                targets = lengths = sample[1][1].to(device)
                loss = lossFunction(outs,targets)
                
                loss.backward()
                optimizer.step()

                lossAbs += float(loss)
                nrBatches += 1
                optimizer.zero_grad()
            print ("EPOCH: {}".format(x+1))
            print ("Loss: {}".format(lossAbs/nrBatches))
            if x % self.evalAll == 0:
                currValLoss = self.validate(model)
                if currValLoss < bestValLoss:
                    bestModel = model
                    bestEpoch = x + 1
                    bestValLoss = currValLoss
        if self.evalAll != 1:
            currValLoss = self.validate(model)
            if currValLoss < bestValLoss:
                bestModel = model
                bestEpoch = self.nrEpochs+1
                bestValLoss = currValLoss
        print ("Best Model in Epoch {} @ {}".format(bestEpoch,bestValLoss))
        return bestModel
    
    def validate(self,model):
        device = model.device
        valData = nl.EyeTrackFullData(self.validationData)
        loader = DataLoader(valData,self.batchSize,num_workers=48,shuffle=True,collate_fn=nl.collate_full_seq,pin_memory=True)
        lossFunction = nn.CrossEntropyLoss()
        with torch.no_grad():
            lossAbs = 0
            nrBatches = 0
            for sample in enumerate(loader):
                ins = sample[1][0].to(device)
                lengths = sample[1][2]
                users = sample[1][3].to(device)

                outs = model(ins,lengths,users)
                targets = lengths = sample[1][1].to(device)

                loss = lossFunction(outs,targets)
                
                lossAbs += float(loss)
                nrBatches += 1
            print ("\tEval Loss: {}".format(lossAbs/nrBatches))
        return lossAbs/nrBatches
    
    def test(self,model):
        device = model.device
        tesData = nl.EyeTrackFullData(self.testData)
        loader = DataLoader(tesData,613,num_workers=48,shuffle=True,collate_fn=nl.collate_full_seq,pin_memory=True)
        model.eval()
        with torch.no_grad():
            for sample in enumerate(loader):
                ins = sample[1][0].to(device)
                lengths = sample[1][2]
                users = sample[1][3].to(device)

                outs = model(ins,lengths,users)
                targets = lengths = sample[1][1].to(device)
                predictions = torch.topk(outs,1).indices.view(-1)
                ###Eval classification
                print ("Test ACC: {}".format((predictions.shape[0]-torch.count_nonzero(torch.sub(predictions,targets)))/predictions.shape[0]))