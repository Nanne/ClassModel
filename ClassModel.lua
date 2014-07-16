-- Implementation of Class Model Visualisation described in:
-- Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. arXiv Preprint arXiv:1312.6034, 1–8. Computer Vision and Pattern Recognition.
--
-- Code partially based on https://github.com/torch/nn/blob/master/Sequential.lua
local ClassModel, Parent = torch.class('nn.ClassModel', 'nn.Sequential')
 
function ClassModel:__init(initial)
    Parent.__init(self)
    if initial:dim() == 3 then
        self.nInputPlane = initial:size(1)
        self.iH = initial:size(2)
        self.iW = initial:size(2)
    elseif initial:dim() == 2 then
        self.nInputPlane = 1
        self.iH = initial:size(1)
        self.iW = initial:size(2)
    else
        error('Need 2 or 3 input dimensions')
    end
    self.weight = initial:clone()
    self.gradWeight = torch.zeros(self.nInputPlane, self.iH, self.iW)
end

function ClassModel:input()
    if torch.Tensor.type(self.weight) == 'torch.CudaTensor' then
        input = self.weight:clone():resize(self.nInputPlane, self.iH, self.iW, 1)
    else
        input = self.weight:clone():resize(1, self.nInputPlane, self.iH, self.iW)
    end
    return input
end

function ClassModel:add(module)
    -- Removing LogSoftMax works better according to Simonyan et al.
    -- Need modified dropout, so we can forwardprop without dropout
    -- and backprop without errors
    local function clean(model)
        for i=1,#model.modules do
            if tostring(model.modules[i]) == 'nn.LogSoftMax' then
                table.remove(model.modules, i)    
            end
            if tostring(model.modules[i]) == 'nn.Dropout' then
                model.modules[i].train = false    
                model.modules[i].updateGradInput = function(input, gradOutput)
                    return model.modules[i].gradInput
                end
            end
        end
    end
    if #self.modules == 0 then
      self.gradInput = module.gradInput
    end
    table.insert(self.modules, module)
    self.output = module.output
    return self
end
 
function ClassModel:updateOutput(input)
    local currentOutput = self:input()
    for i=1,#self.modules do 
       currentOutput = self.modules[i]:updateOutput(currentOutput)
    end 
    self.output = currentOutput
    return currentOutput
end

function ClassModel:parameters()
    return {self.weight}, {self.gradWeight}
end
 
function ClassModel:updateGradInput(input, gradOutput)
    input = self:input()

    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
    local previousModule = self.modules[i]
       currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
       currentModule = previousModule
    end
    currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
    self.gradInput = currentGradOutput
    return currentGradOutput
end

function ClassModel:accGradParameters(input, gradOutput, scale)
    if torch.Tensor.type(self.weight) == 'torch.CudaTensor' then
        input = self.weight:clone():resize(self.nInputPlane, self.iH, self.iW, 1)
    else
        input = self.weight:clone():resize(1, self.nInputPlane, self.iH, self.iW)
    end

    scale = scale or 1

    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
        local previousModule = self.modules[i]
        currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
        currentGradOutput = currentModule.gradInput
        currentModule = previousModule
    end
    currentModule:accGradParameters(input, currentGradOutput, scale)

    self.gradWeight = self.gradWeight:copy(currentModule.gradInput)
    return self.gradWeight
end
 
function ClassModel:reset(initial)
    self.weight = initial:clone()
end

function ClassModel:image()
    im = self.weight:clone()
    im = im:add(-im:min())
    im = im:div(im:max()) 
    return im
end

function ClassModel:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.ClassModel'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
