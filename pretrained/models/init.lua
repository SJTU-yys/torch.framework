--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, modelPath)
   local model
   if modelPath then
      print('=> Loading model: ' .. modelPath)
      model = torch.load(modelPath)
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      print ('parallel')
      model = model:get(1)
   end
   model:remove(#model.modules)
   if opt.featureMap then
      model:remove(#model.modules)
      --model:add(nn.View(2048,4)):setNumInputDims(3)
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      -- Share gradInput for memory efficient backprop
      local cache = {}
      model:apply(function(m)
         local moduleType = torch.type(m)
         if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
            if cache[moduleType] == nil then
               cache[moduleType] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[moduleType], 1, 0)
         end
      end)
      for i, m in ipairs(model:findModules('nn.ConcatTable')) do
         if cache[i % 2] == nil then
            cache[i % 2] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
      end
   end


   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   if opt.nGPU == 0 then
      print ('CPU mode')
      model = model:float()
   end
   return model
end

return M
