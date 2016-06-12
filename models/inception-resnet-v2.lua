local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization


-- Note: 1. Every inception is followed by a 1*1 conv layer for filter-expansion, which is not followed by activation 
--          according to the paper
--       2. There are not bn layer on top of the addition in residue modules because of the large GPU memory consumption
--          it is also mentioned in the paper
local function createModel(opt)
   if opt.nGPU == 0 then
      Convolution = nn.SpatialConvolution
      Avg = nn.SpatialAveragePooling
      ReLU = nn.ReLU
      Max = nn.SpatialMaxPooling
      SBatchNorm = nn.SpatialBatchNormalization
   end
   -- A wrapper of conv + bn + relu layer
   local function ConvBN(ich, och, kW, kH, strideW, strideH, paddingW, paddingH, activation)
       activation = activation or 1
       --print(string.format("conv %d -> %d, kernel (%dx%d), strides (%d, %d), padding (%d, %d)",
       --ich, och, kW, kH, strides[1], strides[2], padding[1], padding[2]))
       local unit = nn.Sequential()
       unit:add(Convolution(ich, och, kW, kH, strideW, strideH, paddingW, paddingH))
       unit:add(SBatchNorm(och))
       if activation == 1 then
          unit:add(ReLU(true))
       end
       return unit
   end
   
   -- stem of inception-resnet-v2 and inception-v4
   local function stem_inception_resnet()  
      stem = nn.Sequential()
      stem:add(ConvBN(3,32,3,3,2,2,0,0))
      stem:add(ConvBN(32,32,3,3,1,1,0,0))
      stem:add(ConvBN(32,64,3,3,1,1,1,1))
   
      local stem_branch1 = ConvBN(64,96,3,3,2,2,0,0)
      local stem_branch2 = nn.Sequential()
      stem_branch2:add(Max(3,3,2,2,0,0))
      local stem_concat1 = nn.Concat(2)
      stem_concat1:add(stem_branch1)
      stem_concat1:add(stem_branch2)
      stem:add(stem_concat1)
   
      local stem_branch3 = nn.Sequential()
      stem_branch3:add(ConvBN(160,64,1,1,1,1,0,0))
      stem_branch3:add(ConvBN(64,64,1,7,1,1,0,3))
      stem_branch3:add(ConvBN(64,64,7,1,1,1,3,0))
      stem_branch3:add(ConvBN(64,96,3,3,1,1,0,0))
      local stem_branch4 = nn.Sequential()
      stem_branch4:add(ConvBN(160,64,1,1,1,1,0,0))
      stem_branch4:add(ConvBN(64,96,3,3,1,1,0,0))
      local stem_concat2 = nn.Concat(2)
      stem_concat2:add(stem_branch3)
      stem_concat2:add(stem_branch4)
      stem:add(stem_concat2)
  
      local stem_concat3 = nn.Concat(2)
      stem_concat3:add(Max(3,3,2,2,0,0))
      stem_concat3:add(ConvBN(192,192,3,3,2,2,0,0))
      stem:add(stem_concat3)

      stem:add(ReLU(true))

      return stem
   end
   
   -- inception module for 35*35 grid of inception-resnet-v2
   local function inception_resnet_A()
      local block = nn.Sequential()
      local inception_branch = nn.Sequential()
   
      -- add inception
      local inception = nn.Concat(2)
      local conv1 = ConvBN(384,32,1,1,1,1,0,0)
      local conv3 = nn.Sequential()
      conv3:add(ConvBN(384,32,1,1,1,1,0,0))
      conv3:add(ConvBN(32,32,3,3,1,1,1,1))
      local conv5 = nn.Sequential()
      conv5:add(ConvBN(384,32,1,1,1,1,0,0))
      conv5:add(ConvBN(32,48,3,3,1,1,1,1))
      conv5:add(ConvBN(48,64,3,3,1,1,1,1))
      inception:add(conv1)
      inception:add(conv3)
      inception:add(conv5)
         
      inception_branch:add(inception)
      inception_branch:add(Convolution(128,384,1,1,1,1,0,0))
      inception_branch:add(nn.MulConstant(0.1))
   
      -- add shortcut and addition
      block:add(nn.ConcatTable()
                :add(nn.Identity())
   	            :add(inception_branch))
      block:add(nn.CAddTable(true))
      block:add(ReLU(true))
      return block
   end
   
   -- reduction module from 35*35 to 17*17
   local function reduction_A()
      local block = nn.Concat(2)
      local max = nn.Sequential()
      max:add(Max(3,3,2,2,0,0))
      local conv3 = ConvBN(384,384,3,3,2,2,0,0)
      local conv5 = nn.Sequential()
      conv5:add(ConvBN(384,256,1,1,1,1,0,0)) 
      conv5:add(ConvBN(256,256,3,3,1,1,1,1)) 
      conv5:add(ConvBN(256,384,3,3,2,2,0,0)) 
      
      block:add(max)
      block:add(conv3)
      block:add(conv5)

      return nn.Sequential()
             :add(block)
   end
    
   -- inception module for 17*17 grid of inception-resnet-v2
   local function inception_resnet_B()
      local block = nn.Sequential()
      local inception_branch = nn.Sequential()

      -- add inception
      local inception = nn.Concat(2)
      local conv1 = ConvBN(1152,192,1,1,1,1,0,0)
      local conv7 = nn.Sequential()
      conv7:add(ConvBN(1152,128,1,1,1,1,0,0))
      conv7:add(ConvBN(128,160,7,1,1,1,3,0))
      conv7:add(ConvBN(160,192,1,7,1,1,0,3))
      inception:add(conv1)
      inception:add(conv7)
   
      inception_branch:add(inception)
      inception_branch:add(Convolution(384,1152,1,1,1,1,0,0))
      inception_branch:add(nn.MulConstant(0.1))

      block:add(nn.ConcatTable()
                :add(nn.Identity())
   	            :add(inception_branch))
      block:add(nn.CAddTable(true))
      block:add(ReLU(true))
      return block
   end
   
   -- reduction module from 17*17 to 8*8
   local function reduction_B()
      local block = nn.Concat(2)
      local max = nn.Sequential()
      max:add(Max(3,3,2,2,0,0))
      local conv3_1 = nn.Sequential()
      conv3_1:add(ConvBN(1152,256,1,1,1,1,0,0))
      conv3_1:add(ConvBN(256,384,3,3,2,2,0,0))
      local conv3_2 = nn.Sequential()
      conv3_2:add(ConvBN(1152,256,1,1,1,1,0,0))
      conv3_2:add(ConvBN(256,288,3,3,2,2,0,0))
      local conv5 = nn.Sequential()
      conv5:add(ConvBN(1152,256,1,1,1,1,0,0))
      conv5:add(ConvBN(256,288,3,3,1,1,1,1))
      conv5:add(ConvBN(288,320,3,3,2,2,0,0))
   
      block:add(max)
      block:add(conv3_1)
      block:add(conv3_2)
      block:add(conv5)
      return nn.Sequential()
             :add(block)
   end
   
   -- inception module for 8*8 grid of inception-resnet-v2
   local function inception_resnet_C()
      local block = nn.Sequential()
      local inception_branch = nn.Sequential()
   
      -- inception branch
      local inception = nn.Concat(2)
      local conv1 = ConvBN(2144,192,1,1,1,1,0,0)
      local conv3 = nn.Sequential()
      conv3:add(ConvBN(2144,192,1,1,1,1,0,0))
      conv3:add(ConvBN(192,224,3,1,1,1,1,0))
      conv3:add(ConvBN(224,256,1,3,1,1,0,1))
      inception:add(conv1)
      inception:add(conv3)

      inception_branch:add(inception)
      inception_branch:add(Convolution(448,2144,1,1,1,1,0,0))
      inception_branch:add(nn.MulConstant(0.1))

      block:add(nn.ConcatTable()
                :add(nn.Identity())
   	            :add(inception_branch))
      block:add(nn.CAddTable(true))
      block:add(ReLU(true))
      return block
   end
   
   -- build model
   local model = nn.Sequential()
   model:add(stem_inception_resnet()) 
   for i =1,5 do
      model:add(inception_resnet_A())
   end
   model:add(reduction_A())
   for i=1,10 do
      model:add(inception_resnet_B())
   end
   model:add(reduction_B())
   for i=1,5 do
      model:add(inception_resnet_C())
   end
   model:add(Avg(8,8,1,1))
   model:add(nn.View(2144):setNumInputDims(3))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(2144,1000))
   --torch.save('inception-resnet-v2.t7',model)
   
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
   torch.save('inception-resnet.t7', model)
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
