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

   local function ConvBN(ich, och, kW, kH, stride, padding)
       local unit = nn.Sequential()
       unit:add(Convolution(ich, och, kW, kH, stride, stride, padding, padding))
       unit:add(SBatchNorm(och))
       unit:add(ReLU(true))
       return unit
   end

   local function inceptionFactoryA(ich, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj)

      local depthConcat = nn.Concat(2)
      local conv1 = ConvBN(ich,num_1x1,1,1,1,0)
      local conv3 = nn.Sequential()
      conv3:add(ConvBN(ich,num_3x3red,1,1,1,0))
      conv3:add(ConvBN(num_3x3red,num_3x3,3,3,1,1))
      local reduce3 = nn.Sequential()
      reduce3:add(ConvBN(ich,num_d3x3red,1,1,1,0))
      reduce3:add(ConvBN(num_d3x3red,num_d3x3,3,3,1,1))
      reduce3:add(ConvBN(num_d3x3,num_d3x3,3,3,1,1))
      local pooling = nn.Sequential()
      if pool == "Avg" then
         pooling:add(Avg(3,3,1,1,1,1))
      else
         pooling:add(Max(3,3,1,1,1,1))
      end
      pooling:add(ConvBN(ich,proj,1,1,1,0))
      depthConcat:add(conv1)
      depthConcat:add(conv3)
      depthConcat:add(reduce3)
      depthConcat:add(pooling)
      
      --return nn.Sequential()
      --         :add(depthConcat)
      return depthConcat
   end

   local function inceptionFactoryB(ich, num_3x3red, num_3x3, num_d3x3red, num_d3x3)
   
      local depthConcat = nn.Concat(2)
      local reduce = nn.Sequential()
      reduce:add(ConvBN(ich,num_3x3red,1,1,1,0))
      reduce:add(ConvBN(num_3x3red,num_3x3,3,3,2,1))
      local doubleReduce = nn.Sequential()
      doubleReduce:add(ConvBN(ich,num_d3x3red,1,1,1,0))
      doubleReduce:add(ConvBN(num_d3x3red,num_d3x3,3,3,1,1))
      doubleReduce:add(ConvBN(num_d3x3,num_d3x3,3,3,2,1))
      local pooling = Max(3,3,2,2,1,1)
      depthConcat:add(reduce)
      depthConcat:add(doubleReduce)
      depthConcat:add(pooling)
   
      --return nn.Sequential()
      --          :add(depthConcat)
      return depthConcat
   end

   local model = nn.Sequential()
   -- STAGE 1
   model:add(ConvBN(3,96,7,7,2,3))
   model:add(Max(3,3,2,2,0,0))
   -- STAGE 2
   model:add(ConvBN(96,128,1,1,1,0))
   model:add(ConvBN(128,288,3,3,1,1))
   model:add(Max(3,3,2,2,0,0))
   -- STAGE 2
   model:add(inceptionFactoryA(288,96,96,96,96,144,"Avg",48))
   model:add(inceptionFactoryA(384,96,96,144,96,144,"Avg",96))
   model:add(inceptionFactoryB(480,192,240,96,144))
   -- STAGE 3
   model:add(inceptionFactoryA(864,224,64,96,96,128,"Avg",128))
   model:add(inceptionFactoryA(576,192,96,128,96,128,"Avg",128))
   model:add(inceptionFactoryA(576,160,128,160,128,160,"Avg",128))
   model:add(inceptionFactoryA(608,96,128,192,160,96,"Avg",128))
   model:add(inceptionFactoryB(512,128,192,192,256))
   -- STAGE 4
   model:add(inceptionFactoryA(960,352,192,320,160,224,"Avg",128))
   model:add(inceptionFactoryA(1024,352,192,320,192,224,"Max",128))
   -- global average pooling
   model:add(Avg(7,7,1,1))
   model:add(nn.View(1024):setNumInputDims(3))
   model:add(nn.Linear(1024,1000))
   
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
