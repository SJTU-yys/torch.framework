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
local t = require 'transforms'

local opt = {}
opt['gen'] = 'gen'
opt['data'] = '/d/home/yushiyao/Place2/256x256'
opt['dataset'] = 'places2'
opt['nThreads'] = 8
opt['batchSize'] = 256

valLoader = DataLoader.create(opt)
-- Load the model
print ('loading model')
local model = torch.load(arg[1])

-- Remove the fully connected layer
--assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local features

function get_feature(model, img_list)
   for i =1,#img_list do
      -- load the image as a RGB float tensor with values 0..1
      -- print (img_list[i])
      local img = image.load(img_list[i], 3, 'float')

      -- Scale, normalize, and crop the image
      img = transform(img)

      -- View as mini-batch of size 1
      img = img:view(1, table.unpack(img:size():totable()))

      -- Get the output of the layer before the (removed) fully connected layer
      local output = model:forward(img:cuda()):squeeze(1)

      if not features then
         features = torch.FloatTensor(#img_list, output:size(1)):zero()
      end

      features[i]:copy(output)
   end
   return features
end

class_list = {}
root = '/d/home/yushiyao/Place2/256x256'
f = io.open(paths.concat(root,"categories.txt"), "r")

imgRoot = root..'/sub_train'
while true do
   local line = f:read('*line')
   if not line then break end
   line = string.split(line, " ")[1]
   table.insert(class_list,string.sub(line, 2))
end

featureRoot = root..'/features'
function process_class(class_list)
   for i=1,#class_list do
      -- process class by class
      local cls = class_list[i]
      print ('processing ',cls)
      -- get image list per class
      local cls_dir = paths.concat(imgRoot,cls)
      -- get class feature
      img_list = {}
      for img in paths.files(paths.concat(cls_dir)) do
         if img ~= '.' and img ~= '..' then
            table.insert(img_list, paths.concat(cls_dir,img))
         end
      end 
      local cls_feature = get_feature(model, img_list)
      -- save the class feature
      feature_path = paths.concat(featureRoot,cls)
      if not paths.dir(feature_path) then
         print ('creating dir ', feature_path)
         paths.mkdir(feature_path)
      end
      torch.save(paths.concat(feature_path,'features.t7'), cls_feature)
      --print('saved features to features.t7')
   end
end
print ('start extracting feature')
process_class(class_list)

--torch.save('features.t7', features)
--print('saved features to features.t7')
