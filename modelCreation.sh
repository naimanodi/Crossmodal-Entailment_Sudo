#!/bin/bash

python3 interface.py -learningRate 0.0001 -gpuNr 0 -feedforwardDim 256 -mode Transformer -batchSize 128 |& tee -a models.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -feedforwardDim 256 -mode Transformerwithoutuser -batchSize 128 |& tee -a models_withoutuser.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -embedDim 32 -mode LSTM -batchSize 128 |& tee -a models.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -embedDim 32 -mode LSTMwithoutuser -batchSize 128 |& tee -a models_withoutuser.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -feedforwardDim 256 -embedDim 32 -mode Ensemble -batchSize 128 |& tee -a models.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -feedforwardDim 256 -embedDim 32 -mode Ensemblewithoutuser -batchSize 128 |& tee -a models_withoutuser.txt
python3 interface.py -learningRate 0.0001 -gpuNr 0 -feedforwardDim 128 -mode BIAS_IND -scalingFactor 500 -batchSize 16 |& tee -a models.txt