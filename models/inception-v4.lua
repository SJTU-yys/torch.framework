local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   
   -- A wrapper of conv + bn + relu layer
   local function Conv(ich, och, kW, kH, strideW, strideH, paddingW, paddingH, activation)
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
   local function stem_inception_v4()  
      local stem = nn.Sequential()
      stem:add(Conv(3,32,3,3,2,2,0,0))
      stem:add(Conv(32,32,3,3,1,1,0,0))
      stem:add(Conv(32,64,3,3,1,1,1,1))
   
      local stem_branch1 = Conv(64,96,3,3,2,2,0,0)
      local stem_branch2 = nn.Sequential()
      stem_branch2:add(Max(3,3,2,2,0,0))
      local stem_concat1 = nn.Concat(2)
      stem_concat1:add(stem_branch1)
      stem_concat1:add(stem_branch2)
      stem:add(stem_concat1)
   
      local stem_branch3 = nn.Sequential()
      stem_branch3:add(Conv(160,64,1,1,1,1,0,0))
      stem_branch3:add(Conv(64,64,1,7,1,1,0,3))
      stem_branch3:add(Conv(64,64,7,1,1,1,3,0))
      stem_branch3:add(Conv(64,96,3,3,1,1,0,0))
      local stem_branch4 = nn.Sequential()
      stem_branch4:add(Conv(160,64,1,1,1,1,0,0))
      stem_branch4:add(Conv(64,96,3,3,1,1,0,0))
      local stem_concat2 = nn.Concat(2)
      stem_concat2:add(stem_branch3)
      stem_concat2:add(stem_branch4)
      stem:add(stem_concat2)
  
      local stem_concat3 = nn.Concat(2)
      stem_concat3:add(Max(3,3,2,2,0,0))
      stem_concat3:add(Conv(192,192,3,3,2,2,0,0))
      stem:add(stem_concat3)

      stem:add(ReLU(true))

      return stem
   end
   
   -- inception module for 35*35 grid of inception-resnet-v2
   local function inception_v4_A()
      local inception = nn.DepthConcat(2)
   
      -- add inception
      local conv_pool = nn.Sequential()
      conv_pool:add(Avg(3,3))
      conv_pool:add(Conv(384,96,1,1,1,1,0,0))
      local conv1 = nn.Sequential()
      conv1:add(Conv(384,96,1,1,1,1,0,0))
      local conv3 = nn.Sequential()
      conv3:add(Conv(384,64,1,1,1,1,0,0))
      conv3:add(Conv(64,96,3,3,1,1,1,1))
      local conv5 = nn.Sequential()
      conv5:add(Conv(384,64,1,1,1,1,0,0))
      conv5:add(Conv(64,96,3,3,1,1,1,1))
      conv5:add(Conv(96,96,3,3,1,1,1,1))
      inception:add(conv_pool)
      inception:add(conv1)
      inception:add(conv3)
      inception:add(conv5)
   
      return inception 
   end
   
   -- reduction module from 35*35 to 17*17
   local function reduction_A()
      local block = nn.DepthConcat(2)
      local max = nn.Sequential()
      max:add(Max(3,3,2,2,0,0))
      local conv3 = Conv(384,384,3,3,2,2,0,0)
      local conv5 = nn.Sequential()
      conv5:add(Conv(384,192,1,1,1,1,0,0)) 
      conv5:add(Conv(192,224,3,3,1,1,1,1)) 
      conv5:add(Conv(224,256,3,3,2,2,0,0)) 
      
      block:add(max)
      block:add(conv3)
      block:add(conv5)
      --block:add(SBatchNorm(1024))
      return block
   end
    
   -- inception module for 17*17 grid of inception-v4
   local function inception_v4_B()
      local inception = nn.DepthConcat(2)

      local conv_pool = nn.Sequential()
      conv_pool:add(Avg(3,3,1,1,1,1))
      conv_pool:add(Conv(1024,128,1,1,1,1,0,0))
      local conv1 = Conv(1024,384,1,1,1,1,0,0)
      local conv7 = nn.Sequential()
      conv7:add(Conv(1024,192,1,1,1,1,0,0))
      conv7:add(Conv(192,224,7,1,1,1,3,0))
      conv7:add(Conv(224,256,1,7,1,1,0,3))
      local conv13 = nn.Sequential()
      conv13:add(Conv(1024,192,1,1,1,1,0,0))
      conv13:add(Conv(192,192,7,1,1,1,3,0))
      conv13:add(Conv(192,224,1,7,1,1,0,3))
      conv13:add(Conv(224,224,7,1,1,1,3,0))
      conv13:add(Conv(224,256,1,7,1,1,0,3))
      inception:add(conv_pool)
      inception:add(conv1)
      inception:add(conv7)
      inception:add(conv13)
      return inception
   end
   
   -- reduction module from 17*17 to 8*8
   local function reduction_B()
      local block = nn.DepthConcat(2)
      local max = nn.Sequential()
      max:add(Max(3,3,2,2,0,0))
      local conv3 = nn.Sequential()
      conv3:add(Conv(1024,192,1,1,1,1,0,0))
      conv3:add(Conv(192,192,3,3,2,2,0,0))
      local conv9 = nn.Sequential()
      conv9:add(Conv(1024,256,1,1,1,1,0,0))
      conv9:add(Conv(256,256,7,1,1,1,3,0))
      conv9:add(Conv(256,320,1,7,1,1,0,3))
      conv9:add(Conv(320,320,3,3,2,2,0,0))
   
      block:add(max)
      block:add(conv3)
      block:add(conv9)
      return block
   end
   
   -- inception module for 8*8 grid of inception-resnet-v2
   local function inception_v4_C()
      local inception = nn.DepthConcat(2)
   
      local conv_pool = nn.Sequential()
      conv_pool:add(Avg(3,3,1,1,1,1))
      conv_pool:add(Conv(1536,256,1,1,1,1,0,0))
      local conv1 = Conv(1536,256,1,1,1,1,0,0)
      local conv3 = nn.Sequential()
      conv3:add(Conv(1536,384,1,1,1,1,0,0))
      conv3:add(nn.DepthConcat(2)
                :add(Conv(384,256,3,1,1,1,1,0))
                :add(Conv(384,256,1,3,1,1,0,1)))
      local conv5 = nn.Sequential()
      conv5:add(Conv(1536,384,1,1,1,1,0,0))
      conv5:add(Conv(384,448,3,1,1,1,1,0))
      conv5:add(Conv(448,512,1,3,1,1,0,1))
      conv5:add(nn.DepthConcat(2)
	        :add(Conv(512,256,1,3,1,1,0,1))
	        :add(Conv(512,256,3,1,1,1,1,0)))
      inception:add(conv_pool)
      inception:add(conv1)
      inception:add(conv3)
      inception:add(conv5)

      return inception
   end
   
   -- build model
   local model = nn.Sequential()
   model:add(stem_inception_v4()) 
   for i =1,4 do
      model:add(inception_v4_A())
   end
   model:add(reduction_A())
   for i=1,7 do
      model:add(inception_v4_B())
   end
   model:add(reduction_B())
   for i=1,3 do
      model:add(inception_v4_C())
   end
   model:add(Avg(8,8,1,1))
   model:add(nn.View(1536):setNumInputDims(3))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(1536,1000))
   --torch.save('inception-v4.t7',model)
   
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
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
