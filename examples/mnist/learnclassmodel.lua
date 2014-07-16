require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'paths'
require 'ClassModel'
require 'dataset-mnist'

----------------------------------------------------------------------
-- parse command-line options
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Class Model learner')
cmd:text()
cmd:text('Options:')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-learningRate', 1e-5, 'learning rate at t=0')
cmd:option('-iterations', 1000, 'number of iterations')
cmd:option('-weightDecay', 1e-5, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:text()
opt = cmd:parse(arg)

if opt.network == '' then
    error('A pre-trained network is required in order to continue')
end

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- because mnist example uses floats

----------------------------------------------------------------------
-- get/create dataset

classes = {'0','1','2','3','4','5','6','7','8','9'}
geometry = {32,32}

nbTrainingPatches = 60000
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)

----------------------------------------------------------------------
-- Get average image
trMean = torch.zeros(geometry[1], geometry[2])
for i=1,trainData:size() do
    trMean = trMean + trainData[i][1]
end
trMean = trMean:div(trainData:size()) 
trMean = trMean:div(trMean:max()) -- scale maximal value to 1

----------------------------------------------------------------------
-- Load existing model
model = torch.load(opt.network)

----------------------------------------------------------------------
-- Save the average image (scaled between 0 and 1)
os.execute('mkdir images')
image.save('images/mean.png', image.minmax{tensor=trMean, min=0, max=1})

----------------------------------------------------------------------
--Initialise model with average image
cmodel = nn.ClassModel(trMean)
-- in Simonyan paper the model is initialised with all 0's and then the mean is added to the final image.
-- Both options seem to work for mnist.
cmodel:add(model)

print('<Class Model> using model:')
print(cmodel)

config = config or {learningRate = opt.learningRate,
                      weightDecay = opt.weightDecay,
                      momentum = opt.momentum,
                      learningRateDecay = 5e-7}

criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- Train class model
for t = 1, #classes do
    target = torch.zeros(1)
    target[1] = t

    cmodel:reset(trMean:clone())
    parameters,gradParameters = cmodel:getParameters()
    print('Generating class model image of class ' .. classes[t])

    for e = 1,opt.iterations do
        xlua.progress(e, opt.iterations)

        local feval = function(x)
            local output = cmodel:forward({})

            local f = criterion:forward(output, target)

            local df_do = criterion:backward(output, target)

            cmodel:backward({}, df_do)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, config)
    end
    -- save class model image
    image.save('images/' .. classes[t] .. '.png', cmodel:image())
end
