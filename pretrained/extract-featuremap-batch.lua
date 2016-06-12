--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

require 'torch'
require 'paths'
require 'io'
local DataLoader = require 'dataloader'

--if #arg < 1 then
--   io.stderr:write('Usage: th extract-features-batch.lua [MODEL]...\n')
--   os.exit(1)
--end
for _, f in ipairs(arg) do
   if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
   end
end

require 'cudnn'
require 'cunn'
require 'image'
local models = require 'models/init'
--local t = require 'transforms'

local opt = {}
opt['gen'] = 'gen'
opt['data'] = '/d/home/yushiyao/Place2/256x256/val'
opt['dataset'] = 'places2'
opt['nThreads'] = 8
opt['batchSize'] = 256
opt['nGPU'] = 4
opt['shareGradInput'] = true
opt['netType'] = 'resnet'
opt['featureMap'] = true
opt['avgType'] = 'arith'

-- Load the model
--print ('loading model')
--local model = models.setup(opt, arg[1])
local model = models.setup(opt, arg[1])
--model:add(nn.View(50,2048,4)):setNumInputDims(2)
--if opt.avgType == 'arith' then
--   model:add(nn.Mean(3)):add(nn.Mean(4))
--end
-- local model = torch.load(arg[1])

-- Remove the fully connected layer
--assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
-- model:remove(#model.modules)

-- Setup the dataloader
local loader = DataLoader.create(opt)
-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

--local transform = t.Compose{
--   t.Scale(256),
--   t.ColorNormalize(meanstd),
--   t.CenterCrop(224),
--}

local features

--function get_feature(model, img_list)
--   for i =1,#img_list do
--      -- load the image as a RGB float tensor with values 0..1
--      -- print (img_list[i])
--      local img = image.load(img_list[i], 3, 'float')
--
--      -- Scale, normalize, and crop the image
--      img = transform(img)
--
--      -- View as mini-batch of size 1
--      img = img:view(1, table.unpack(img:size():totable()))
--
--      -- Get the output of the layer before the (removed) fully connected layer
--      local output = model:forward(img:cuda()):squeeze(1)
--
--      if not features then
--         features = torch.FloatTensor(#img_list, output:size(1)):zero()
--      end
--
--      features[i]:copy(output)
--   end
--   return features
--end

function process_dataset(dataloader)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()
   print ('Number of batches: ', size)
   local feature = {}
   local classIdx = {}
   for i =1,401 do
      feature[i] = {}
      feature[i]['feature'] = torch.Tensor(50,2048,2,2)
      feature[i]['name'] = {}
      classIdx[i] = 1
   end
   

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      input, target = copyInputs(sample)
      -- print (input:size(), target:size())
      
      local output = model:forward(input):float()
      print (output:size())
      -- print (n)
      --print (n,size, timer:time().real, dataTime)
      print (('Extract:[%d/%d] Time:%.3f Data:%.3f'):format(n, size, timer:time().real, dataTime))
      timer:reset()
      dataTimer:reset()
      local idx 
      for t = 1, input:size()[1] do
         idx = sample.target[t]
         --print (feature[idx]:size())
         --print (feature[idx][classIdx[idx]]:size())
         feature[idx]['feature'][classIdx[idx]]:resize(output[t]:size())
         --if opt.avgType == 'geo' then
         --   for f =1,4 do
         --      output[t]
         --   end
         --end
         feature[idx]['feature'][classIdx[idx]] = output[t]
         feature[idx]['name'][classIdx[idx]] = sample.name[t]
         classIdx[idx] = classIdx[idx] + 1
      end
   end
   --torch.save('features.t7', features)
   --print ('whole feature saving complete!')
   print ('start saving features')
   for j =1,401 do
      feature_path = paths.concat('/d/home/yushiyao/Place2/256x256/val_feature_map',j)
      if not paths.dir(feature_path) then
         print ('creating dir ', feature_path)
         paths.mkdir(feature_path)
      end
      torch.save(paths.concat(feature_path, 'features_'..j..'.t7'), feature[j])
   end
end
function copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   local input = torch.FloatTensor()
   local target = torch.FloatTensor()
   input:resize(sample.input:size()):copy(sample.input)
   target:resize(sample.target:size()):copy(sample.target)
   return input, target
end

print ('start extracting feature')
process_dataset(loader)

--torch.save('features.t7', features)
--print('saved features to features.t7')
