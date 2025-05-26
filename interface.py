import torch
import argparse
from datetime import datetime
import model as nm
import trainer as nt
from transformers.utils import logging

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-learningRate', default=0.00004, type=float, help="Learning rate")
    parser.add_argument('-gpuNr', default= 0, type=int, help="Which Gpu to use")
    parser.add_argument('-embedDim', default= 16, type=int, help="User Embedding dimension")
    parser.add_argument('-feedforwardDim',default=512,type=int, help="Dimension of Transformer FeedForward")
    parser.add_argument('-batchSize', default=32, type=int,help="Size of batches")
    parser.add_argument('-mode', default="pretrain", type=str, help="(pretrain|finetune|crossvalidation|nobias|biasdecode")
    parser.add_argument('-saveAs', default="", type=str, help="save as name")
    parser.add_argument('-scalingFactor',default=500,type=int,help="Bias scaling factor")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
    logging.set_verbosity_error()
    print (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    args = get_parameter()
    print ("Mode: {}\nFF: {}\nEmb: {}\nBatch: {}\nLR: {}\nScaling: {}".format(args.mode,args.feedforwardDim,args.embedDim,args.batchSize,args.learningRate,args.scalingFactor))

    if args.mode == "LSTM":
        model = nm.LSTMSequence(args.embedDim,args.gpuNr,True)
        trainer = nt.LSTMTrainer(args.batchSize,30,args.learningRate,True)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"LSTM_Model{}.pt".format(args.saveAs))
    elif args.mode == "LSTMwithoutuser":
        model = nm.LSTMSequence(args.embedDim,args.gpuNr,False)
        trainer = nt.LSTMTrainer(args.batchSize,30,args.learningRate,False)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"LSTM_Model_withoutuser{}.pt".format(args.saveAs))
    elif args.mode == "Transformer":
        model = nm.TransformerModel(6,6,args.feedforwardDim,0.2,args.gpuNr,True)
        trainer = nt.IndividualSequenceTrainer(args.batchSize,30,args.learningRate,True)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Transformer_Model{}.pt".format(args.saveAs))
    elif args.mode == "Transformerwithoutuser":
        model = nm.TransformerModel(6,6,args.feedforwardDim,0.2,args.gpuNr,False)
        trainer = nt.IndividualSequenceTrainer(args.batchSize,30,args.learningRate,False)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Transformer_Model_withoutuser{}.pt".format(args.saveAs))
    elif args.mode == "Ensemble":
        model = nm.Ensemble(args.embedDim,6,6,args.feedforwardDim,0.2,args.gpuNr,True,True)
        trainer = nt.EnsembleTrainer(args.batchSize,30,args.learningRate,True)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Ensemble_Model(avg){}.pt".format(args.saveAs))
        model = nm.Ensemble(args.embedDim,6,6,args.feedforwardDim,0.2,args.gpuNr,False,True)
        trainer = nt.EnsembleTrainer(args.batchSize,30,args.learningRate,True)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Ensemble_Model(learned){}.pt".format(args.saveAs))
    elif args.mode == "Ensemblewithoutuser":
        model = nm.Ensemble(args.embedDim,6,6,args.feedforwardDim,0.2,args.gpuNr,True,False)
        trainer = nt.EnsembleTrainer(args.batchSize,30,args.learningRate,False)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Ensemble_Model_withoutuser(avg){}.pt".format(args.saveAs))
        model = nm.Ensemble(args.embedDim,6,6,args.feedforwardDim,0.2,args.gpuNr,False,False)
        trainer = nt.EnsembleTrainer(args.batchSize,30,args.learningRate,False)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Ensemble_Model_withoutuser(learned){}.pt".format(args.saveAs))
    elif args.mode == "BIAS_IND":
        model = nm.BiasTestModel(6,6,args.feedforwardDim,0.2,args.gpuNr)
        trainer = nt.IndividualSequenceTrainerBias(args.batchSize,30,args.learningRate,args.scalingFactor)
        best = trainer.train(model)
        trainer.test(best)
        torch.save(best,"Transformer_Bias_Model{}.pt".format(args.saveAs))
    print (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))