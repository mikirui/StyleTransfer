require 'torch'
require 'hdf5'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	assert(opt.h5_file, 'h5_file path not provided')
	assert(opt.batch_size, 'batch_size not provided')

	self.h5_file = hdf5.open(opt.h5_file, 'r')
	self.batch_size = opt.batch_size
	self.split_idxs = {
		train = 1,
		val = 1,
	}
	self.image_path = {
		train = '/train2014/images',
		val = 'val2014/images',
	}

	local train_size = self.h5_file:read(self.image_path.train):dataspaceSize()
	print("train data size: ")
	print(train_size)

	self.split_sizes = {
		train = train_size[1],
		val = self.h5_file:read(self.image_path.val):dataspaceSize()[1],
	}
	self.channels = train_size[2]
	self.height = train_size[3]
	self.width = train_size[4]

	if opt.max_train and opt.max_train > 0 then
		self.split_sizes.train = opt.max_train
	end
	
	self.num_batches = {}
	for k, v in pairs(self.split_sizes) do 
		self.num_batches[k] = math.floor(v / self.batch_size)
	end

	self.rgb2gray = torch.FloatTensor{0.2989, 0.5870, 0.1140}
end

function DataLoader:reset(split)
	self.split_idxs[split] = 1
end

function DataLoader:getBatch(split)
	local path = self.image_path[split]
	local start_idx = self.split_idxs[split]
	local end_idx = math.min(start_idx + self.batch_size - 1, self.split_sizes[split])

	local images = self.h5_file:read(path):partial(
		{start_idx, end_idx},
		{1, self.channels},
		{1, self.height},
		{1, self.width}):float():div(255)
	self.split_idxs[split] = end_idx + 1
	if self.split_idxs[split] > self.split_sizes[split] then
		self.split_idxs[split] = 1
	end

	--preprocess before feed in the network
	images_proc = preprocess_batch(images)
	return images_proc
end

function preprocess_batch(img_batch)
	assert(img_batch:dim() == 4, 'img_batch should be N x C x H xW')
	assert(img_batch:size(2) == 3, 'images should have 3 channels')

	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3,2,1}
	img_batch = img_batch:index(2, perm):mul(255.0)
	mean_pixel = mean_pixel:view(1, 3, 1, 1):expandAs(img_batch)
	img_batch = img_batch:add(-1, mean_pixel)
	return img_batch
end 

function deprocess_batch(img_batch)
	assert(img_batch:dim() == 4, 'img_batch should be N x C x H xW')
	assert(img_batch:size(2) == 3, 'images should have 3 channels')

	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	mean_pixel = mean_pixel:view(1, 3, 1, 1):expandAs(img_batch)
	img_batch = img_batch + mean_pixel
	local perm = torch.LongTensor{3,2,1}
	img_batch = img_batch:index(2, perm):div(255.0)
	return img_batch
end

