import torch

import torch.nn as nn
import torch.nn.utils.rnn as rnn

from transformers import BertForMaskedLM

class BiasTestModel(nn.Module):

    def __init__(self,nrHeads,nrLayers,feedforwardDim,dropout,gpuNr = 0):
        super(BiasTestModel,self).__init__()
        ###Variables
        self.feedforwardDim = feedforwardDim
        self.inputDimension = 768
        self.nrHeads = nrHeads
        self.nrLayers = nrLayers
        self.gpuNr = gpuNr
        self.dropout = dropout
        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')
        ###Layer definition
        #bert for inputIds
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased",output_hidden_states=True).to(self.device)
        for params in self.bert.parameters():
            params.requires_grad = False
        #img to word dimension
        self.downsize = nn.Linear(2048,768).to(self.device)
        torch.nn.init.xavier_uniform_(self.downsize.weight)
        #multi-modal transformer
        self.encoderLayer = nn.TransformerEncoderLayer(self.inputDimension,self.nrHeads,self.feedforwardDim,dropout=self.dropout,batch_first=True).to(self.device)
        self.transformerEncoder = nn.TransformerEncoder(self.encoderLayer,self.nrLayers).to(self.device)
        for x in range(self.nrLayers):
            torch.nn.init.xavier_uniform_(self.transformerEncoder.layers[x].linear1.weight)
            torch.nn.init.xavier_uniform_(self.transformerEncoder.layers[x].linear2.weight)

        self.activation = nn.ReLU().to(self.device)
        self.usrEmb = nn.Embedding(106,16).to(self.device)
        torch.nn.init.xavier_uniform_(self.usrEmb.weight)
        #does caption fit (human evaluation!)
        self.classifierSecTask = nn.Linear(self.inputDimension+16,3).to(self.device)
        torch.nn.init.xavier_uniform_(self.classifierSecTask.weight)

    def getDevice(self):
        return self.device

    def forward(self,inputs):
        #img features
        downsizedImgs = self.downsize(inputs[0])
        downsizedImgs = self.activation(downsizedImgs)
        #word features
        wordFeatures = self.bert(**inputs[2])
        wordFeatures = wordFeatures['hidden_states'][-1]
        #build input
        mmInput = torch.cat((wordFeatures,downsizedImgs),1)
        encoderMask = torch.cat((inputs[3],inputs[1]),1)

        srcMask = inputs[4].repeat(self.nrHeads,1,1)
        mmRepresentations = self.transformerEncoder(mmInput,src_key_padding_mask=encoderMask,mask=srcMask)

        usr = self.usrEmb(inputs[-1])
        clsClassification = self.classifierSecTask(torch.cat((mmRepresentations[:,0],usr.squeeze()),dim=1))
        return clsClassification

class TransformerModel(nn.Module):
    def __init__(self, nrHeads, nrLayers, feedforwardDim, dropout, gpuNr=0,userembed=True):
        super(TransformerModel, self).__init__()
        
        # Variables
        self.feedforwardDim = feedforwardDim
        self.inputDimension = 768
        self.nrHeads = nrHeads
        self.nrLayers = nrLayers
        self.gpuNr = gpuNr
        self.dropout = dropout
        self.device = torch.device('cuda:' + str(self.gpuNr) if torch.cuda.is_available() else 'cpu')

        # BERT backbone
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).to(self.device)
        for params in self.bert.parameters():
            params.requires_grad = False

        # Linear downsize for image features
        self.downsize = nn.Linear(2048, 768).to(self.device)
        torch.nn.init.xavier_uniform_(self.downsize.weight)

        # Transformer encoder
        self.encoderLayer = nn.TransformerEncoderLayer(self.inputDimension, self.nrHeads, self.feedforwardDim, dropout=self.dropout, batch_first=True).to(self.device)
        self.transformerEncoder = nn.TransformerEncoder(self.encoderLayer, self.nrLayers).to(self.device)
        for x in range(self.nrLayers):
            torch.nn.init.xavier_uniform_(self.transformerEncoder.layers[x].linear1.weight)
            torch.nn.init.xavier_uniform_(self.transformerEncoder.layers[x].linear2.weight)

        self.activation = nn.ReLU().to(self.device)

        # User embedding
        self.usrEmb = nn.Embedding(106, 16).to(self.device)
        torch.nn.init.xavier_uniform_(self.usrEmb.weight)

        # Classification head
        self.classifierSecTask = nn.Linear(self.inputDimension + 16, 3).to(self.device)
        torch.nn.init.xavier_uniform_(self.classifierSecTask.weight)

        # Toggle flag
        self.use_user_embeddings = userembed

    def getDevice(self):
        return self.device

    def forward(self, inputs):
        # inputs = (imageFeatures, imageMask, bertInputs, bertMask, userIds)
        user_ids = inputs[-1]
        wordInputs = inputs[2]
        imgFeatures = inputs[0]
        imgMask = inputs[1]
        bertMask = inputs[3]

        downsizedImgs = self.activation(self.downsize(imgFeatures))
        wordFeatures = self.bert(**wordInputs)['hidden_states'][-1]

        mmInput = torch.cat((wordFeatures, downsizedImgs), dim=1)
        encoderMask = torch.cat((bertMask, imgMask), dim=1)
        mmRepresentations = self.transformerEncoder(mmInput, src_key_padding_mask=encoderMask)

        if self.use_user_embeddings:
            usr = self.usrEmb(user_ids).squeeze()
            clsClassification = self.classifierSecTask(torch.cat((mmRepresentations[:, 0], usr), dim=1))
        else:
            clsClassification = self.classifierSecTask(torch.cat((mmRepresentations[:, 0], torch.zeros((mmRepresentations.shape[0], 16), device=self.device)), dim=1))

        return clsClassification

class LSTMSequence(nn.Module):
    def __init__(self, embSize, gpuNr=0,userembed=True):
        super(LSTMSequence, self).__init__()
        self.embSize = embSize
        self.gpuNr = gpuNr

        self.device = torch.device('cuda:' + str(self.gpuNr) if torch.cuda.is_available() else 'cpu')

        self.userEmbed = nn.Embedding(num_embeddings=107, embedding_dim=self.embSize).to(self.device)
        torch.nn.init.xavier_uniform_(self.userEmbed.weight)

        self.embed = nn.Embedding(num_embeddings=837, embedding_dim=self.embSize, padding_idx=0).to(self.device)
        torch.nn.init.xavier_uniform_(self.embed.weight)
    
        self.lstm = nn.LSTM(input_size=self.embSize, hidden_size=self.embSize).to(self.device) #old code
        '''
        self.lstm = nn.LSTM(
            input_size=self.embSize,
            hidden_size=self.embSize,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        ).to(self.device)
        '''
        self.classifier = nn.Linear(self.embSize, 3).to(self.device)
        #self.classifier = nn.Linear(self.embSize * 2, 3).to(self.device) 

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        
        self.representationTransformation = nn.Linear(self.embSize * 2, self.embSize).to(self.device)
        torch.nn.init.xavier_uniform_(self.representationTransformation.weight)

        '''
        #Normalization and dropout post LSTM
        self.layer_norm = nn.LayerNorm(self.embSize * 2).to(self.device)  # for bidirectional
        self.post_lstm_dropout = nn.Dropout(0.5)
        '''
        self.activation = nn.ReLU()

        # New flag for controlling user embeddings
        self.use_user_embeddings = userembed

    def getDevice(self):
        return self.device

    def forward(self, inputs, lengths, users):
        embeds = self.embed(inputs)

        if self.use_user_embeddings:
            userEmb = self.userEmbed(users)
            inputrepresentation = torch.cat((embeds, userEmb.expand(embeds.shape)), dim=2)
            inputrepresentation = self.activation(self.representationTransformation(inputrepresentation))
            packedInputs = rnn.pack_padded_sequence(inputrepresentation, lengths, enforce_sorted=False)
            rnnouts, (hn, cn) = self.lstm(packedInputs, (userEmb, userEmb))
        else:
            inputrepresentation = embeds
            packedInputs = rnn.pack_padded_sequence(inputrepresentation, lengths, enforce_sorted=False)
            rnnouts, (hn, cn) = self.lstm(packedInputs)

        outs = self.classifier(cn[-1])
        return outs
        '''
        if self.use_user_embeddings:
            userEmb = self.userEmbed(users)
            inputrepresentation = torch.cat((embeds, userEmb.expand(embeds.shape)), dim=2)
            inputrepresentation = self.activation(self.representationTransformation(inputrepresentation))
        else:
            inputrepresentation = embeds

        packedInputs = rnn.pack_padded_sequence(inputrepresentation, lengths, enforce_sorted=False)
        packedOutputs, (hn, cn) = self.lstm(packedInputs)
        lstm_out, _ = rnn.pad_packed_sequence(packedOutputs, batch_first=True)

        # Apply normalization + dropout
        normed = self.layer_norm(lstm_out)
        dropped = self.post_lstm_dropout(normed)

        # Use the last time step of both directions
        # (batch_size, 2*hidden_size)
        final_out = torch.cat((hn[-2], hn[-1]), dim=1)

        return self.classifier(final_out)
        '''
class Ensemble(nn.Module):
    def __init__(self,embSize,nrHeads,nrLayers,feedforwardDim,dropout,gpuNr = 0,f=True,userembed=True):
        super(Ensemble,self).__init__()
        self.userembed=userembed
        self.sequenceModel = LSTMSequence(embSize,gpuNr,self.userembed)
        self.featureModel = TransformerModel(nrHeads,nrLayers,feedforwardDim,dropout,gpuNr,self.userembed)
        self.device = self.featureModel.device
        self.finalDecision = nn.Linear(6,3).to(self.device)
        self.dropout = nn.Dropout(0.05)
        self.f=f
    
    def forward(self,seqIns,featIns):
        sequences,lengths,users = seqIns
        model1Outs = self.sequenceModel(sequences,lengths,users)
        model2Outs = self.featureModel(featIns)
        if self.f:
            outs = torch.add(model1Outs,model2Outs)/2
        else:
            combined = torch.cat((model1Outs, model2Outs), dim=1)
            outs = self.finalDecision(combined)
        return outs


