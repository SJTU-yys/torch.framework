--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   if opt.nGPU == 0 then
      Convolution = nn.SpatialConvolution
      Avg = nn.SpatialAveragePooling
      ReLU = nn.ReLU
      Max = nn.SpatialMaxPooling
      SBatchNorm = nn.SpatialBatchNormalization
   end
  
   local features = nn.Sequential()
   features:add(Convolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   -- 55 ->  27
   features:add(Convolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   --  27 ->  13
   features:add(Convolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU(true))
   features:add(Convolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU(true))
   features:add(Convolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6):setNumInputDims(3))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
   -- classifier:add(nn.LogSoftMax())
   --
   -- features:get(1).gradInput = nil

   local model = nn.Sequential()
   model:add(features):add(classifier)

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if opt.nGPU > 0 then
            if cudnn.version >= 4000 then
               v.bias = nil
               v.gradBias = nil
            else
               v.bias:zero()
            end
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         if opt.nGPU > 0 then
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
