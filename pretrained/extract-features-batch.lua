--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model

require 'torch'
require 'paths'
require 'io'
local DataLoader = require 'dataloader'

if #arg < 1 then
   io.stderr:write('Usage: th extract-features-batch.lua [MODEL]...\n')
   os.exit(1)
end
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

local opt = {}
opt.gen = 'gen'
opt.data = '/d/home/yushiyao/Place2/256x256/val'
opt.dataset = 'places2'
opt.nThreads = 8
opt.batchSize = 128
opt.nGPU = 2
opt.shareGradInput = true
opt.featureMap = false
opt.outputDir = '/d/home/yushiyao/Place2/'

-- Load the model
print ('loading model')
local model = models.setup(opt, arg[1])

-- Setup the dataloader
local loader = DataLoader.create(opt)
-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local features

function process_dataset(dataloader)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()
   print ('Number of batches: ', size)
   local feature = {}
   feature['name'] = {}
   
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      input = copyInputs(sample)
      
      local output = model:forward(input):float()
      print (('Extract:[%d/%d] Time:%.3f Data:%.3f'):format(n, size, timer:time().real, dataTime))
      timer:reset()
      dataTimer:reset()
      local idx 
      if n == 1 then
         feature['feature'] = output
      else
         feature['feature'] = torch.cat(feature['feature'], output, 1)
         print (feature['feature']:size())
      end
      for i,name in ipairs(sample.name) do
         table.insert(feature['name'],name)
      end
   end

   print ('start saving features')
   feature_path = paths.concat(opt.outputDir)
   if not paths.dir(feature_path) then
      print ('creating dir ', feature_path)
      paths.mkdir(feature_path)
   end
   torch.save(paths.concat(feature_path,'feature.t7'), feature)
end

function copyInputs(sample)
   local input = (opt.nGPU == 1
          and torch.CudaTensor()
          or cutorch.createCudaHostTensor())
   input:resize(sample.input:size()):copy(sample.input)
   return input
end

print ('start extracting feature')
process_dataset(loader)
