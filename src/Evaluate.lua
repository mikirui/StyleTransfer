require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'InstanceNormalization'
require 'Cut'
require 'Dataloader'
local cjson = require 'cjson'

model_path = '../log/checkpoint2.t7'
--model_json = '../log/checkpoint2.json'
img_path = '../images/content/tubingen.jpg'
out_path = '../images/output/oo3.jpg'
img_max_size = -1

dtype = 'torch.FloatTensor'

model = torch.load(model_path).model
model = model:type(dtype)
model:evaluate()

local img = image.load(img_path, 3, 'float')
if(img_max_size ~= -1) then
	img = image.scale(img, img_max_size, 'bilinear')
end

local C,H,W = img:size(1), img:size(2), img:size(3)
img = img:view(1,C,H,W)
local img_proc = preprocess_batch(img):type(dtype)
print(img_proc:size())
print("forward img")
local res = model:forward(img_proc)
torch.save('../log/res.t7', res)

local disp = deprocess_batch(res:float()):float()
local N,C,H,W = disp:size(1), disp:size(2), disp:size(3), disp:size(4)
disp = disp:view(C, H, W)        
torch.save('../log/disp.t7', disp)
disp = image.minmax{tensor=disp, min=0, max=1}  --[0,255] ->[0,1]
image.save(out_path, disp)
