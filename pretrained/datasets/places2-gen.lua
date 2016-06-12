--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of Places2 filenames and classes
--
--  This generates a file gen/places2.t7 which contains the list of all
--  Places2 training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local io = require 'io'
local string = require 'string'

local M = {}


local function findImages(dir)
   local imagePath = torch.CharTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      --local className = paths.basename(paths.dirname(line))
      local dirName = paths.dirname(line)
      local filename = paths.basename(line)
      local path = dirName .. '/' .. filename

      table.insert(imagePaths, path)
      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   return imagePath
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset

   --local trainDir = paths.concat(opt.data, 'train')
   --local valDir = paths.concat(opt.data, 'val')
   local dataDir = opt.data
   --local classFileDir = paths.concat(opt.data, 'categories.txt')
   --assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   --assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   assert(paths.dirp(dataDir), 'data directiry not found:'  .. dataDir)

   print(" | finding all images")
   local dataImagePath = findImages(dataDir)

   local info = {
      basedir = opt.data,
      imagePath = dataImagePath,
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
