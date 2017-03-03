require 'torch'
require 'nn'
require 'InstanceNormalization'
require 'Cut'

local function Normalization(dim, norm_type)
	if norm_type == 'batch' then 
		return nn.SpatialBatchNormalization(dim)
	elseif norm_type == 'instance' then
		return nn.InstanceNormalization(dim)
	else
		error(string.format("No such normalization type: %s", norm_type))
	end
end

local function ResBlock(dim, norm_type)
	local res_block = nn.Sequential()

	--building the conv block
	local conv_block = nn.Sequential()
	conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1))
	conv_block:add(Normalization(dim, norm_type))
	conv_block:add(nn.ReLU(true))
	conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1))
	conv_block:add(Normalization(dim, norm_type))

	--connect to form res_block
	local concat = nn.ConcatTable()
	concat:add(conv_block)
	concat:add(nn.Cut(2))
	res_block:add(concat):add(nn.CAddTable())
	return res_block
end

local function ConvBlock(conv_type, norm_type, activate, i_dim, o_dim, filter, stride, pad, adj)
	local conv_block = nn.Sequential()
	if conv_type == 'Down' then
		conv_block:add(nn.SpatialConvolution(i_dim, o_dim, filter, filter, stride, stride, pad, pad))
	else
		conv_block:add(nn.SpatialFullConvolution(i_dim, o_dim, filter, filter, stride, stride, pad, pad, adj, adj))
	end

	conv_block:add(Normalization(o_dim, norm_type))
	if activate == 'Tanh' then
		conv_block:add(nn.Tanh())
	else
		conv_block:add(nn.ReLU(true))
	end
	return conv_block
end

function TransformNet(norm_type)
	local tf_net = nn.Sequential()                    --e.g. input = [3,256,256]
	--tf_net:add(nn.SpatialReflectionPadding(40,40,40,40))  --[3,336,336]
	tf_net:add(ConvBlock('Down', norm_type, 'ReLU', 3, 32, 9, 1, 4))  --[32,336,336]
	tf_net:add(ConvBlock('Down', norm_type, 'ReLU', 32, 64, 3, 2, 1))  --[64, 168, 168]
	tf_net:add(ConvBlock('Down', norm_type, 'ReLU', 64, 128, 3, 2, 1))  --[128, 84,84]

	--Res_block
	local Res_block_num = 5                 --[128,84,84] -> [128,64,64]
	for i =1, Res_block_num do
		tf_net:add(ResBlock(128, norm_type))
	end

	tf_net:add(ConvBlock('Up', norm_type, 'ReLU', 128, 64, 3, 2, 1, 1))   --[64,128,128]
	tf_net:add(ConvBlock('Up', norm_type, 'ReLU', 64, 32, 3, 2, 1, 1))     --[32, 256, 256]
	tf_net:add(ConvBlock('Down', norm_type, 'Tanh', 32, 3, 9, 1, 4))      --[3, 256, 256]
	return tf_net
end

