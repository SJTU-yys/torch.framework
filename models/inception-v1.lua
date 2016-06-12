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

   local function inception(depth_dim, input_size, config)
   
      local depth_concat = nn.Concat(depth_dim)
      local conv1 = nn.Sequential()
      conv1:add(Convolution(input_size, config[1][1], 1, 1)):add(ReLU(true))
      depth_concat:add(conv1)
   
      local conv3 = nn.Sequential()
      conv3:add(Convolution(input_size, config[2][1], 1, 1)):add(ReLU(true))
      conv3:add(Convolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)):add(ReLU(true))
      depth_concat:add(conv3)
   
      local conv5 = nn.Sequential()
      conv5:add(Convolution(input_size, config[3][1], 1, 1)):add(ReLU(true))
      conv5:add(Convolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)):add(ReLU(true))
      depth_concat:add(conv5)
   
      local pool = nn.Sequential()
      pool:add(Max(config[4][1], config[4][1], 1, 1, 1, 1))
      pool:add(Convolution(input_size, config[4][2], 1, 1)):add(ReLU(true))
      depth_concat:add(pool)
   
      return depth_concat
   end

   local model = nn.Sequential()
   model:add(Convolution(3,64,7,7,2,2,3,3)):add(ReLU(true))
   model:add(Max(3,3,2,2,1,1))
   -- LRN (not added for now)
   model:add(Convolution(64,64,1,1,1,1,0,0)):add(ReLU(true))
   model:add(Convolution(64,192,3,3,1,1,1,1)):add(ReLU(true))
   -- LRN (not added for now)
   model:add(Max(3,3,2,2,1,1))
   model:add(inception(2, 192, {{ 64}, { 96,128}, {16, 32}, {3, 32}})) -- 256
   model:add(inception(2, 256, {{128}, {128,192}, {32, 96}, {3, 64}})) -- 480
   model:add(Max(3,3,2,2,1,1))
   model:add(inception(2, 480, {{192}, { 96,208}, {16, 48}, {3, 64}})) -- 4(a)
   model:add(inception(2, 512, {{160}, {112,224}, {24, 64}, {3, 64}})) -- 4(b)
   model:add(inception(2, 512, {{128}, {128,256}, {24, 64}, {3, 64}})) -- 4(c)
   model:add(inception(2, 512, {{112}, {144,288}, {32, 64}, {3, 64}})) -- 4(d)
   model:add(inception(2, 528, {{256}, {160,320}, {32,128}, {3,128}})) -- 4(e) (14x14x832)
   model:add(Max(3,3,2,2,1,1))
   model:add(inception(2, 832, {{256}, {160,320}, {32,128}, {3,128}})) -- 5(a)
   model:add(inception(2, 832, {{384}, {192,384}, {48,128}, {3,128}})) -- 5(b)
   model:add(Avg(7,7,1,1))
   model:add(nn.View(1024):setNumInputDims(3))
   -- model:add(nn.Dropout(0.4))
   model:add(nn.Linear(1024,1000)):add(nn.ReLU(true))
   -- model:add(nn.LogSoftMax())
   
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
