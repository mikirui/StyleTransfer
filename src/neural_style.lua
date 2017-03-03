require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'

local cmd = torch.CmdLine()

--Basic options
cmd:option('-content_image','../images/content/tubingen.jpg', 'Content target image')
cmd:option('-style_image', '../images/style/seated-nude.jpg', 'Style target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-multigpu_strategy', '', 'Index of layers to split the network across GPUs')

--Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('normalize_gradients', false)    --?
cmd:option('-init', 'random', 'random|image')
cmd:option('-init_image', '')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)
cmd:option('-lbfgs_num_correction', 0)     --?

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

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')


local function main(params)
	local dtype, multigpu = setup_gpu(params)
	local loadcaffe_backend = params.backend
	assert(params.backend == 'nn' or params.backend == 'cudnn', 'only nn|cudnn are provided for backend')
	--load pre-trained model
	print("loading pretrained cnn")
	local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)

	--load content images & style images
	print("loading content images and style images")
	local content_image = image.load(params.content_image, 3)
	content_image = image.scale(content_image, params.image_size, 'bilinear')
	local content_image_proc = preprocess(content_image):float()

	local style_size = math.ceil(params.style_scale * params.image_size)
	local style_image = image.load(params.style_image, 3)
	style_image = image.scale(style_image, style_size, 'bilinear')
	local style_image_proc = preprocess(style_image):float()

	--Get content and style loss layers, set up the network
	print("Get content and style loss layers, set up the network...")
	local content_layers = params.content_layers:split(",")
	local style_layers = params.style_layers:split(",")

	local content_losses, style_losses = {}, {}
	local content_idx, style_idx = 1,1
	local net = nn.Sequential()
	if params.tv_weight > 0 then
		local tv_mod = nn.TVLoss(params.tv_weight):type(dtype)
		net:add(tv_mod)
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
				net:add(avg_pool)
			else
				net:add(layer)
			end
			--For content loss layers
			if layer_name == content_layers[content_idx] then
				print("Setting up content layer ", i, ":", layer.name)
				local norm = params.normalize_gradients
				local loss_mod = nn.ContentLoss(params.content_weight, norm):type(dtype)
				net:add(loss_mod)
				table.insert(content_losses, loss_mod)
				content_idx = content_idx + 1
			end
			--For style loss layers
			if layer_name == style_layers[style_idx] then
				print("Setting up style layer ", i, ":", layer.name)
				local norm = params.normalize_gradients
				local loss_mod = nn.StyleLoss(params.style_weight, norm):type(dtype)
				net:add(loss_mod)
				table.insert(style_losses, loss_mod)
				style_idx = style_idx + 1
			end
		end
	end
	if multigpu then 
		--net = setup_multi_gpu(net, params)
	end
	net:type(dtype)
	print('network architecture: ')
	print(net)

	--Capture content and style loss layer feature maps, since they are static
	for i = 1, #content_losses do         
		content_losses[i].mode = 'Capture'   --Forward will not update image x   
	end
	print('capturing content images feature maps')
	content_image_proc = content_image_proc:type(dtype)
	net:forward(content_image_proc)
	for i = 1, #content_losses do
		content_losses[i].mode = 'none'   --reset
	end

	for i = 1,#style_losses do
		style_losses[i].mode = 'Capture'
	end
	print('capturing style images feature maps')
	style_image_proc = style_image_proc:type(dtype)
	net:forward(style_image_proc)

	--Set all loss modules to loss mode
	for i = 1, #content_losses do
		content_losses[i].mode = 'loss'
	end
	for i = 1, #style_losses do
		style_losses[i].mode = 'loss'
	end

	--Release cnn and remove unnecessary grad weights
	cnn = nil
	for i = 1, #net.modules do
		local mod = net.modules[i]
		if torch.type(mod) == 'nn.SpatialConvolutionMM' then  
			--weights in theses layer would not be updated
			mod.gradWeight = nil
			mod.gradBias = nil
		end
	end
	collectgarbage()

	--Init image with random noise or other images(content img or style img)
	if params.seed >=0 then
		torch.manualSeed(params.seed)
	end
	local init_image = nil
	if params.init == 'random' then
		init_image = torch.randn(content_image:size()):float():mul(0.001)
	elseif params.init == 'image' then
		if params.init_image ~= '' then
			init_image = image.load(params.init_image, 3)
			init_image = image.scale(content_image, init_image, 'bilinear')  --size fit content img
			init_image = preprocess(init_image):float()
		else
			init_image = content_image_proc:clone()  --default content img
		end
	else
		error('Invalid init type')
	end
	init_image = init_image:type(dtype)

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
			learning_rate = params.learning_rate,
		}
	else
		error(string.format('Unrecognized optimizer: %s', params.optimizer))
	end

	local function maybe_print(t,loss)
		local verbose = (params.print_iter > 0 and t %params.print_iter == 0)
		if verbose then
			print(string.format('Iteration %d / %d ', t, params.num_iterations))
			for i, loss_mod in ipairs(content_losses) do
				print(string.format('Content %d loss %f', i, loss_mod.loss))
			end
			for i,loss_mod in ipairs(style_losses) do
				print(string.format('Style %d loss: %f', i, loss_mod.loss))
			end
			print(string.format('Total loss: %f', loss))
		end
	end

	local function maybe_save(t)
		local should_save = (params.save_iter > 0 and t % params.save_iter == 0)
		should_save = should_save or t == params.num_iterations      --save the last iteration result
		if should_save then
			local disp = deprocess(init_image:double())         --deprocess to show
			disp = image.minmax{tensor=disp, min=0, max=1}  --[0,255] ->[0,1]
			local filename = build_filename(params.output_image, t)
			if t == params.num_iterations then
				filename = params.output_image   --"out.jpg"
			end

			--color-indepentdent style transfer
			if params.original_colors == 1 then
				disp = original_colors(content_image, disp)
			end
			image.save(filename, disp)
		end
	end

	local y = net:forward(init_image)
	local dy = init_image.new(#y):zero()   --backward with loss 0 from last layer (new?)
	local iter = 0
	local function feval(x)
		iter = iter + 1
		net:forward(x)
		local grad = net:updateGradInput(x, dy)    --get 
		local loss = 0
		for _, mod in ipairs(content_losses) do
			loss = loss + mod.loss 
		end
		for _,mod in ipairs(style_losses) do
			loss = loss + mod.loss 
		end
		maybe_print(iter, loss)
		maybe_save(iter)
		collectgarbage()
		return loss, grad:view(grad:nElement())
	end

	--Run optimization
	if params.optimizer == 'lbfgs' then
		print('Running optimization with L-BFGS')
		local x, loss = optim.lbfgs(feval, init_image, optim_state)
	elseif params.optimizer == 'adam' then
		print('Running optimization with ADAM')
		for t = 1, params.num_iterations do 
			local x, loss = optim.adam(feval, init_image, optim_state)
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
	local dtype = 'torch.FLoatTensor'
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
	local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3,2,1}
	img = img:index(1, perm):mul(255.0)
	mean_pixel = mean_pixel:view(3,1,1):expandAs(img)
	img:add(-1, mean_pixel)
	return img
end

--depreprocess
function deprocess(img)
	local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
	mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
	img = img + mean_pixel
	local perm = torch.LongTensor{3,2,1}
	img = img:index(1, perm):div(256.0)
	return img
end

function original_colors(content, generated)
	local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
	local content_uv = image.rgb2yuv(content)[{{2,3}}]
	return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


--ContentLoss Module
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(weight, normalize)
	parent.__init(self)
	self.weight = weight
	self.target = torch.Tensor()
	self.normalize = normalize or false
	self.loss = 0
	self.crit = nn.MSECriterion()
	self.mode = 'none'
end

function ContentLoss:updateOutput(input)
	if self.mode == 'loss' then
		self.loss = self.crit:forward(input, self.target) * self.weight
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
		self.gradInput:mul(self.weight)
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
end

function Gram:updateOutput(input)
	assert(input:dim() == 3)
	local C, H, W = input:size(1), input:size(2), input:size(3)
	local x_flat = input:view(C, H*W)
	self.output:resize(C, C)
	self.output:mm(x_flat, x_flat:t())
	return self.output
end

function Gram:updateGradInput(input, gradOutput)    --?
	assert(input:dim() == 3 and input:size(1))
	local C, H, W = input:size(1), input:size(2), input:size(3)
	local x_flat = input:view(C, H*W)
	self.gradInput:resize(C, H*W):mm(gradOutput, x_flat)
	self.gradInput:addmm(gradOutput:t(), x_flat)
	self.gradInput = self.gradInput:view(C, H, W)
	return self.gradInput
end


--StyleLoss Module
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')
function StyleLoss:__init(weight, normalize)
	parent.__init(self)
	self.normalize = normalize or false
	self.weight = weight
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
		self.loss = self.weight * self.crit:forward(self.G, self.target)
	end
	self.output = input
	return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
	if self.mode == 'loss' then
		local dG = self.crit:backward(self.G, self.target)
		dG:div(input:nElement())
		self.gradInput = self.gram:backward(input, dG)
		if self.normalize then
			self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
		end
		self.gradInput:mul(self.weight)
		self.gradInput:add(gradOutput)
	else
		self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	end
	return self.gradInput
end


--TVLoss Module
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(weight)
	parent.__init(self)
	self.weight = weight
	self.x_diff = torch.Tensor()
	self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
	self.output = input
	return self.output
end

function TVLoss:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input):zero()
	local C, H, W = input:size(1), input:size(2), input:size(3)
	self.x_diff:resize(3, H-1, W-1)
	self.y_diff:resize(3, H-1, W-1)
	self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
	self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
	self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
	self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
	self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
	self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
	self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
	self.gradInput:mul(self.weight)
	self.gradInput:add(gradOutput)
	return self.gradInput
end

local params = cmd:parse(arg)
main(params)

