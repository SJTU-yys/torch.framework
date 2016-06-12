#!/bin/sh
#th main.lua -depth 101 -batchSize 256 -nGPU 4 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012/ -gen res101
#export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1
#th main.lua -netType resnet -depth 50 -dataset imagenet -batchSize 32 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012 -testOnly true -retrain tmp/fc2conv_resnet-200.t7
#th main.lua -netType resnet -depth 50 -dataset places2 -batchSize 64 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/places365/standard_384 -testOnly true -retrain fc2conv_places365_standard384_resnet50_model_43.t7
#th main.lua -netType resnet_pre -depth 200 -dataset imagenet -batchSize 128 -nGPU 4 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012 -testOnly true -retrain resnet200_79.t7
#th main.lua -netType inception-v4 -dataset imagenet -batchSize 32 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012 -trainingStyle layerwise
#th main.lua -netType inception-resnet-v2 -dataset imagenet -batchSize 32 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012 
#th main.lua -netType resnet -depth 50 -dataset imagenet -batchSize 128 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012  
th main.lua -netType resnet -depth 50 -dataset places2 -batchSize 16 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/places365/standard_384 -subdata 500 
#th main.lua -netType preresnet -depth 200 -dataset imagenet -batchSize 16 -nGPU 2 -nThreads 8 -shareGradInput true -data /d/data/ILSVRC2012 
#th main.lua -netType resnet -depth 50 -dataset places2 -batchSize 128 -nGPU 2 -nThreads 16 -shareGradInput true -data /d/data/places365/standard_384 -testOnly true -retrain places365_standard384_resnet101_model_42.t7 -record false -tenCrop true
#th main.lua -netType resnet -depth 50 -dataset places2 -batchSize 2 -nGPU 1 -nThreads 1 -shareGradInput true -data /d/data/Places365/standard_384 -testOnly true -retrain places365_standard384_resnet101_model_42.t7 -record true -tenCrop true
