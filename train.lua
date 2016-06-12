--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = false,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5
      lossSum = lossSum + loss
      N = N + 1

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   local predFile
   local confusionMatrix
   if self.opt.record == 'true' then
      predFile = io.open(opt.predictionFile, 'w')
      confusionMatrix = torch.Tensor(365,365):fill(0)
   end
   
   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      --local myoutput = output
      --myoutput = myoutput:view(output:size(1) / nCrops, nCrops, output:size(2))
      if nCrops > 1 then
         output = output:view(output:size(1)/nCrops, nCrops, output:size(2))
      end

      -- get the predictions
      local m = nn.SoftMax()
      if self.opt.record == 'true' then
         for i =1,output:size()[1] do
            -- calculate mean prob first
            tmpProb = m:forward(output[i]):mean(1)
            --tmpProb = torch.mean(tmpProb, 1)
            local prob, predictions = tmpProb:float():sort(2,true)
            prob = prob:squeeze():mul(100)
            predictions = predictions:squeeze()
            -- get single sample
            local item = dataloader.dataset:getPath(sample.indices[i])
            -- find where is the ground truth in the prediction vector
            local targetIndex = -1
            for k = 1,predictions:size()[1] do
               if predictions[k] == item.target then
                  targetIndex = k
               end
            end
            confusionMatrix[item.target] = confusionMatrix[item.target] + prob
            predFile:write(('%s %d %d %d %d %d %d %d %d %d %d %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'):format(item.img, item.target, targetIndex, predictions[1], predictions[2], predictions[3], predictions[4], predictions[5], predictions[6], predictions[7], predictions[8], predictions[9], predictions[10], prob[1], prob[2], prob[3], prob[4], prob[5], prob[6], prob[7], prob[8], prob[9], prob[10]))
         end
      end

      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5
      N = N + 1

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   if self.opt.record then
      torch.save(opt.confusionMatrixFile, confusionMatrix,'ascii')
   end

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   -- Computes the top1 and top5 error rate, if use multi-crop, calculate the average of the output probability
   -- and get the argmax
   local newOutput
   local batchSize = output:size(1)
   if nCrops > 1 then
      local size = output:size():totable()
      table.remove(size,2)
      newOutput = torch.FloatTensor(table.unpack(size))
      local m = nn.SoftMax()
      for i =1,output:size(1) do
          -- calculate mean prob first
          newOutput[i] = m:forward(output[i]):mean(1)
      end
   else 
      newOutput = output
   end

   local _ , predictions = newOutput:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize,1):expandAs(newOutput))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   if self.opt.nGPU == 0 then
      self.input = self.input or torch.FloatTensor()
      self.target = self.target or torch.FloatTensor()
   else
      self.input = self.input or (self.opt.nGPU == 1
          and torch.CudaTensor()
          or cutorch.createCudaHostTensor())
      self.target = self.target or torch.CudaTensor()
   end
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      if epoch == 1 then
         return 0.01
      else
         decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      end
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
