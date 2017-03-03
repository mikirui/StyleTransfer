require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'
require 'TransformNet'
require 'Dataloader'

local cjson = require 'cjson'

local cmd = torch.CmdLine()

--Basic options
cmd:option('-content_image','../images/content/tubingen.jpg', 'Content target image')
cmd:option('-style_image', '../images/style/seated-nude.jpg', 'Style target image')
cmd:option('-image_size', 256, 'Maximum height / width of generated image')
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-multigpu_strategy', '', 'Index of layers to split the network across GPUs')

--Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('normalize_gradients', false)    --?
cmd:option('-optimizer', 'adam', 'lbfgs|adam|sgd')
cmd:option('-learning_rate', 1e1)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-lr_decay_every', -1)
cmd:option('-lbfgs_num_correction', 0)     --?

--DataLoader options
cmd:option('-h5_file', '/dev_data/wrz/mscoco_256.h5')
cmd:option('-batch_size', 4)
cmd:option('-max_train', -1, 'max index of trainset to train')

--Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', '../images/output/out.jpg')

--Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)    --?
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', '../models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', '../models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-cudnn_autotune', false)   --?
cmd:option('-seed', -1)
cmd:option('-normalization', 'instance', 'instance|batch')
cmd:option('-num_val_batches', 100, 'number of batches to evaluate')
cmd:option('-checkpoint_name', 'checkpoint')

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')


local function main(params)
	local dtype, multigpu = setup_gpu(params)
	local loadcaffe_backend = params.backend
	assert(params.backend == 'nn' or params.backend == 'cudnn', 'only nn|cudnn are provided for backend')

	--building Image Transform Network
	TFNet = TransformNet(params.normalization)
	if params.backend == 'cudnn' then 
		cudnn.convert(TFNet, cudnn)
	end

	--load pre-trained model
	print("loading pretrained cnn")
	local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)

	--Get content and style loss layers, set up the network
	print("Get content and style loss layers, set up the network...")
	local content_layers = params.content_layers:split(",")
	local style_layers = params.style_layers:split(",")

	local content_losses, style_losses, tv_losses = {}, {}, {}
	local content_idx, style_idx = 1,1
	local LossNet = nn.Sequential()
	if params.tv_weight > 0 then
		local tv_mod = nn.TVLoss(params.tv_weight):type(dtype)
		table.insert(tv_losses, tv_mod)
		LossNet:add(tv_mod)
	end

	--Access each layer of pretrained layed
	for i = 1, #cnn do 
		--print(string.format("content_idx:%d, style_idx: %d",content_idx, style_idx))
		if content_idx <= #content_layers or style_idx <= #style_layers then
			local layer = cnn:get(i)
			local layer_name = layer.name 
			local layer_type = torch.type(layer)
			--Max_pool -> Avg_pool
			local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
			if is_pooling and params.pooling == 'avg' then 
				assert(layer.padW == 0 and layer.padH ==0, 'padding should be 0')
				local kW, kH = layer.kW, layer.kH
				local dW, dH = layer.dW, layer.dH
				local avg_pool = nn.SpatialAveragePooling(kW, kH, dW,dH):type(dtype)
				print(string.format('replacing max_pooling at layer %d with avg_pooling', i))
				LossNet:add(avg_pool)
			else
				LossNet:add(layer)
			end
			--For content loss layers
			if layer_name == content_layers[content_idx] then
				print("Setting up content layer ", i, ":", layer.name)
				local norm = params.normalize_gradients
				local loss_mod = nn.ContentLoss(params.content_weight, norm):type(dtype)
				LossNet:add(loss_mod)
				table.insert(content_losses, loss_mod)
				content_idx = content_idx + 1
			end
			--For style loss layers
			if layer_name == style_layers[style_idx] then
				print("Setting up style layer ", i, ":", layer.name)
				local norm = params.normalize_gradients
				local loss_mod = nn.StyleLoss(params.style_weight, norm):type(dtype)
				LossNet:add(loss_mod)
				table.insert(style_losses, loss_mod)
				style_idx = style_idx + 1
			end
		end
	end
	if multigpu then 
		--net = setup_multi_gpu(net, params)
	end

	--Full net
	FullNet = nn.Sequential()
	FullNet:add(TFNet)
	FullNet:add(LossNet)
	FullNet = FullNet:type(dtype)
	if params.backend == 'cudnn' then 
		cudnn.convert(FullNet, cudnn)
	end
	FullNet:training()
	print('network architecture: ')
	print(FullNet)

	--load style image
	print("loading style images")
	local style_size = math.ceil(params.style_scale * params.image_size)
	local style_image = image.load(params.style_image, 3, 'float')
	style_image = image.scale(style_image, style_size, 'bilinear')
	local img_C, img_H, img_W = style_image:size(1), style_image:size(2), style_image:size(3)
	style_image=  style_image:view(1, img_C, img_H, img_W)
	local style_image_proc = preprocess_batch(style_image):float()
	--Capture style loss layer feature maps, since they are static
	for i = 1,#style_losses do
		style_losses[i].mode = 'Capture'
	end
	print('capturing style images feature maps')
	style_image_proc = style_image_proc:type(dtype)
	LossNet:forward(style_image_proc)                --forward to get the style loss
	for i = 1, #style_losses do
		style_losses[i].mode = 'loss'
	end

	--Release cnn and remove unnecessary grad weights
	print("Release cnn...")
	cnn = nil
	for i = 1, #LossNet.modules do
		local mod = LossNet.modules[i]
		if torch.type(mod) == 'cudnn.SpatialConvolution' then  
			--weights in theses layer would not be updated
			print(string.format("disaable %d layer in the vgg",i))
			mod.gradWeight = nil
			mod.gradBias = nil
		end
	end
	collectgarbage()

	--Optimization setting
	local optim_state = nil
	if params.optimizer == 'lbfgs' then 
		optim_state = {
			maxIter = params.num_iterations,   --iterate specific steps
			verbose = true,
			tolX = -1,
			tolFun = -1,
		}
		if params.lbfgs_num_correction > 0 then 
			optim_state.nCorrection = params.lbfgs_num_correction
		end
	elseif params.optimizer == 'adam' then 
		optim_state = {
			learningRate = params.learning_rate,
		}
	else
		error(string.format('Unrecognized optimizer: %s', params.optimizer))
	end

	--prepare for the dataset
	local loader = DataLoader(params)
	par, gradParams = TFNet:getParameters()
	local rand_img = torch.rand(params.batch_size, 3, params.image_size, params.image_size)
	rand_img = nn.SpatialReflectionPadding(40,40,40,40):forward(rand_img):type(dtype)
	local y = FullNet:forward(rand_img)
	local dy = rand_img.new(#y):zero():type(dtype)
	print(par:size())
	print(gradParams:size())

	local function maybe_print(t,loss)              --print the losses
		local verbose = (params.print_iter > 0 and t %params.print_iter == 0)
		local epoch = t / loader.num_batches['train']
		if verbose then
			print(string.format('Epoch %d, Iteration %d / %d ', epoch, t, params.num_iterations))
			for i, loss_mod in ipairs(content_losses) do
				print(string.format('Content %d loss %f', i, loss_mod.loss))
			end
			for i,loss_mod in ipairs(style_losses) do
				print(string.format('Style %d loss: %f', i, loss_mod.loss))
			end
			for i,loss_mod in ipairs(tv_losses) do
				print(string.format('tv loss %d: %f', i, loss_mod.loss))
			end
			print(string.format('Total loss: %f', loss))
		end
	end

	--checkpoint option
	local train_loss_history = {}
	local val_loss_history = {}
	local val_loss_history_ts = {}
	local style_loss_history = {}
	for i,loss_mod in ipairs(style_losses) do
		style_loss_history[string.format('style-%d', i)] = {}
	end
	for i,loss_mod in ipairs(content_losses) do
		style_loss_history[string.format('content-%d', i)] = {}
	end

	local function maybe_save(t, loss)                        --save the checkpoint
		local should_save = (params.save_iter > 0 and t % params.save_iter == 0)
		should_save = should_save or t == params.num_iterations      --save the last iteration result
		if should_save then
			--record the history 
			table.insert(train_loss_history, loss)
			for i,loss_mod in ipairs(style_losses) do
				table.insert(style_loss_history[string.format('style-%d', i)], loss_mod.loss)
			end
			for i,loss_mod in ipairs(content_losses) do
				table.insert(style_loss_history[string.format('content-%d', i)], loss_mod.loss)
			end
			loader:reset('val')
			FullNet:evaluate()

			--Validate on valset
			local val_loss = 0
			print 'Running on validation set ...'
			local num_batches = params.num_val_batches
			for j = 1, num_batches do
				local batchInput = loader:getBatch('val'):double()
				local batchInput_pad = nn.SpatialReflectionPadding(40,40,40,40):forward(batchInput)
				batchInput = batchInput:type(dtype)
				batchInput_pad = batchInput_pad:type(dtype)

				for i = 1, #style_losses do
					style_losses[i].mode = 'none'
				end
				for i = 1, #content_losses do         
					content_losses[i].mode = 'Capture'   --Forward will not update image x   
				end
				LossNet:forward(batchInput)    --Capture ContentLoss layer feature
				for i = 1, #content_losses do
					content_losses[i].mode = 'loss'   --set to loss mode
				end
				for i = 1, #style_losses do
					style_losses[i].mode = 'loss'   --set to loss mode
				end
				FullNet:forward(batchInput_pad)
				for _, mod in ipairs(content_losses) do
					val_loss = val_loss + mod.loss 
				end
				for _,mod in ipairs(style_losses) do
					val_loss = val_loss + mod.loss 
				end
				for _,mod in ipairs(tv_losses) do
					val_loss = val_loss + mod.loss 
				end
			end
			val_loss = val_loss / num_batches
			print(string.format('val_loss: %f', val_loss))
			table.insert(val_loss_history, val_loss)
			table.insert(val_loss_history_ts, t)

			FullNet:training()
			local checkpoint = {
				params = params,
				train_loss_history = train_loss_history,
				val_loss_history = val_loss_history,
				val_loss_history_ts = val_loss_history_ts,
				style_loss_history = style_loss_history,
			}
			local filename = string.format('%s_%d.json', params.checkpoint_name, t)
			paths.mkdir(paths.dirname(filename))
			write_json(filename, checkpoint)

			--Save a torch checkpoint, convert model to float
			FullNet:clearState()
			if params.backend == 'cudnn' then
				cudnn.convert(FullNet, nn)
			end
			FullNet:float()
			checkpoint.model = FullNet
			filename = string.format('%s_%d.json', params.checkpoint_name, t)
			torch.save(filename, checkpoint)

			--convert the model back
			FullNet:type(dtype)
			if params.backend == 'cudnn' then
				cudnn.convert(FullNet, cudnn)
			end
			par, gradParams = TFNet:getParameters()
		end
	end

	local iter = 0
	local function feval(par)
		--fetch data
		local batchInput = loader:getBatch('train'):double()
		local batchInput_pad = nn.SpatialReflectionPadding(40,40,40,40):forward(batchInput)
		batchInput = batchInput:type(dtype)
		batchInput_pad = batchInput_pad:type(dtype)

		--Capture the contentLoss layer feature
		iter = iter + 1
		for i = 1, #style_losses do
			style_losses[i].mode = 'none'
		end
		for i = 1, #content_losses do         
			content_losses[i].mode = 'Capture'   --Forward will not update image x   
		end
		print('Capturing content images feature maps')
		LossNet:forward(batchInput)    --Capture ContentLoss layer feature
		for i = 1, #content_losses do
			content_losses[i].mode = 'loss'   --set to loss mode
		end
		for i = 1, #style_losses do
			style_losses[i].mode = 'loss'   --set to loss mode
		end

		--calculate loss and grad
		gradParams:zero()
		print("tfnet forward...")
		local mid = TFNet:forward(batchInput_pad)
		print("lossnet forward...")
		LossNet:forward(mid)
		print("tfnet backward...")
		local grad_mid = LossNet:updateGradInput(mid, dy)
		print("lossnet backward...")
		print(TFNet)
		TFNet:backward(batchInput_pad, grad_mid)
		print("finish")
		local loss = 0
		for _, mod in ipairs(content_losses) do
			loss = loss + mod.loss 
		end
		for _,mod in ipairs(style_losses) do
			loss = loss + mod.loss 
		end
		for _,mod in ipairs(tv_losses) do
			loss = loss + mod.loss 
		end
		maybe_print(iter, loss)
		maybe_save(iter, loss)
		if params.lr_decay_every > 0 and iter % params.lr_decay_every == 0 then
			local new_lr = params.lr_decay_factor * optim_state.learningRate
			optim_state = {learningRate = new_lr}
		end
		collectgarbage()
		return loss, gradParams
	end

	--Run optimization
	if params.optimizer == 'lbfgs' then
		print('Running optimization with L-BFGS')
		local x, loss = optim.lbfgs(feval, init_image, optim_state)
	elseif params.optimizer == 'adam' then
		print('Running optimization with ADAM')
		for t = 1, params.num_iterations do 
			local x, loss = optim.adam(feval, par, optim_state)
		end
	end
end

--other function
function setup_gpu(params)
	local multigpu = false
	if params.gpu:find(',') then 
		multigpu = true
		params.gpu = params.gpu:split(',')
		for i = 1, #params.gpu do
			params.gpu[i] = tonumber(params.gpu[i]) + 1
		end
	else
		params.gpu = tonumber(params.gpu) + 1
	end
	local dtype = 'torch.FloatTensor'
	if multigpu or params.gpu > 0 then
		if params.backend == 'cudnn' then
			require 'cutorch'
			require 'cunn'
			if multigpu then
				cutorch.setDevice(params.gpu[1])
			else
				cutorch.setDevice(params.gpu)
			end
			dtype = 'torch.CudaTensor'
		else
			error('set gpu without cudnn backend')
		end
	else
		params.backend = 'nn'
	end
	if params.backend == 'cudnn' then
		require 'cudnn'
		if params.cudnn_autotune then
			cudnn.benchmark = true    --?
		end
		cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters
	end
	return dtype, multigpu
end

function build_filename(output_image, iteration)
	local ext = paths.extname(output_image)
	local basename = paths.basename(output_image, ext)
	local dir = paths.dirname(output_image)
	return string.format("%s/%s_%d.%s", dir, basename, iteration, ext)
end

--[0,1] -> [0,255], RGB->BGR
function preprocess(img)
	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3,2,1}
	img = img:index(1, perm):mul(255.0)
	mean_pixel = mean_pixel:view(3,1,1):expandAs(img)
	img:add(-1, mean_pixel)
	return img
end

--depreprocess
function deprocess(img)
	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
	img = img + mean_pixel
	local perm = torch.LongTensor{3,2,1}
	img = img:index(1, perm):div(256.0)
	return img
end

function write_json(path, cont)
	cjson.encode_sparse_array(true, 2, 10)
	local text = cjson.encode(cont)
	local file = io.open(path, 'w')
	file:write(text)
	file:close()
end

function original_colors(content, generated)
	local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
	local content_uv = image.rgb2yuv(content)[{{2,3}}]
	return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end

--ContentLoss Module
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
	parent.__init(self)
	self.strength = strength
	self.target = torch.Tensor()
	self.normalize = normalize or false
	self.loss = 0
	self.crit = nn.MSECriterion()
	self.mode = 'none'
end

function ContentLoss:updateOutput(input)
	if self.mode == 'loss' then
		self.loss = self.crit:forward(input, self.target) * self.strength
	elseif self.mode == 'Capture' then
		self.target:resizeAs(input):copy(input)
	end
	self.output = input
	return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
	if self.mode == 'loss' then
		if input:nElement() == self.target:nElement() then
			self.gradInput = self.crit:backward(input, self.target)
		end
		if self.normalize then
			self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
		end
		self.gradInput:mul(self.strength)
		self.gradInput:add(gradOutput)
	else
		self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	end
	return self.gradInput
end

--GramMatrix Module
local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')

function Gram:__init()
	parent.__init(self)
	self.buffer = torch.Tensor()
end

function Gram:updateOutput(input)
	local C, H, W
	if input:dim() == 3 then 
		C, H, W = input:size(1), input:size(2), input:size(3)
		local x_flat = input:view(C, H*W)
		self.output:resize(C, C)
		self.output:mm(x_flat, x_flat:t())
	elseif input:dim() == 4 then
		local N = input:size(1)
		C, H, W = input:size(2), input:size(3), input:size(4)
		local x_flat = input:view(N, C, H*W)
		self.output:resize(N, C, C)
		self.output:bmm(x_flat, x_flat:transpose(2, 3))
	end
	self.output:div(C * H * W)   --normalize
	return self.output
end

function Gram:updateGradInput(input, gradOutput)    --?
	self.gradInput:resizeAs(input):zero()
	local C, H, W
	if input:dim() == 3 then
		C, H, W = input:size(1), input:size(2), input:size(3)
		local x_flat = input:view(C, H*W)
		self.buffer:resize(C, H*W):mm(gradOutput, x_flat)
		self.buffer:addmm(gradOutput:t(), x_flat)
		self.gradInput = self.buffer:view(C, H, W)
	elseif input:dim() == 4 then
		local N = input:size(1)
		C, H, W = input:size(2), input:size(3), input:size(4)
		local x_flat = input:view(N, C, H*W)
		self.buffer:resize(N, C, H*W):bmm(gradOutput, x_flat)
		self.buffer:baddbmm(gradOutput:transpose(2, 3), x_flat)
		self.gradInput = self.buffer:view(N, C, H, W)
	end
	self.buffer:div(C * H *W)
	assert(self.gradInput:isContiguous())
	return self.gradInput
end


--StyleLoss Module
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')
function StyleLoss:__init(strength, normalize)
	parent.__init(self)
	self.normalize = normalize or false
	self.strength = strength
	self.target = torch.Tensor()
	self.mode = 'none'
	self.loss = 0

	self.gram = nn.GramMatrix()
	self.G = nil
	self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
	self.G = self.gram:forward(input)
	self.G:div(input:nElement())
	if self.mode == 'Capture' then
		self.target:resizeAs(self.G):copy(self.G)
	elseif self.mode == 'loss' then
		local target = self.target
		if self.G:size(1) > 1 and self.target:size(1) == 1 then
			target = target:expandAs(self.G)
		end
		self.loss = self.strength * self.crit:forward(self.G, target)
		self._target = target
	end
	self.output = input
	return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
	if self.mode == 'loss' then
		local dG = self.crit:backward(self.G, self._target)
		self.gradInput = self.gram:backward(input, dG)
		if self.normalize then
			self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
		end
		self.gradInput:mul(self.strength)
		self.gradInput:add(gradOutput)
	else
		self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	end
	return self.gradInput
end


--TVLoss Module
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
	parent.__init(self)
	self.strength = strength
	self.x_diff = torch.Tensor()
	self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
	self.output = input
	return self.output
end

function TVLoss:updateGradInput(input, gradOutput)
	assert(input:dim() == 4, "input should be N x C x H x W")
	self.gradInput:resizeAs(input):zero()
	local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
	self.x_diff:resize(N, 3, H-1, W-1)
	self.y_diff:resize(N, 3, H-1, W-1)
	self.x_diff:copy(input[{ {}, {}, {1, -2}, {1, -2}}])
	self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
	self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
	self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
	self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
	self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
	self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
	self.gradInput:mul(self.strength)
	self.gradInput:add(gradOutput)
	return self.gradInput
end

local params = cmd:parse(arg)
main(params)

