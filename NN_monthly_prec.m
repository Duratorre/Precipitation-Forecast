clear all
close all

%addpath C:\Users\Duratorre\Documents\MATLAB\NeuralNetwork\

% load precipitation data from Tokyo weather station
load monthly

%% plot autocorrelation
nlag = 200;
x = [0:nlag];
for i=1:nlag+1
    varianza = (-1+1.645*sqrt(size(prep,1)-x-1))./(size(prep,1)-x);
end

figure(1)
autocorr(prep,nlag,[],0);
title('Autocorrelation function - Precipitation data')
xlabel('Lag [s]')
hold on
grid on 
grid minor
plot(x,varianza,'b')
plot(x,-varianza,'b')
legend('correlogram','upper conf. bond','lower conf. bond','Location','best')

%% define starting indices for partitioning training, validation and test subsets
init = 264;
trainind = 204;
valind = 252;
testind = 264;

% plot the precipitation data
figure(2)
plot(time(end-init+1:end), prep(end-init+1:end))
grid on, grid  minor
xlim([time(end-init+1) time(end)]) 

%% prepare input and output for the neural network
ttime = time(end-init+1:end)-time(end-init+1);
inputn = (ttime')./(numel(time(end-init+1:end-(init-trainind)))+1);
% outputn =  (prep(end-init+1:end)-min(prep(end-init+1:end)))./(max(prep(end-init+1:end))-min(prep(end-init+1:end)));
outputn = prep(end-init+1:end)./max(prep(end-init+1:end)); 
[input, wasMatrix] = tonndata(inputn, false, false);
[output, wasMatriy] = tonndata(outputn, false, false);

%% define neural network architecture
net = network;
net.numInputs = 1;
net.numLayers = 5;
net.inputs{1}.size = 1;
net.inputConnect(1,1) = 1;
net.inputConnect(2,1) = 1;
net.inputConnect(3,1) = 1;
net.inputConnect(4,1) = 1;
net.layers{1}.size = 250;
net.layers{2}.size = 4;
net.layers{3}.size = 4;
net.layers{4}.size = 4;
net.layers{5}.size = 1;
net.layerConnect(5,1) = 1;
net.layerConnect(5,2) = 1;
net.layerConnect(5,3) = 1;
net.layerConnect(5,4) = 1;
net.outputConnect(5) = 1;
net.biasConnect(1) = 1;
net.biasConnect(2) = 1;
net.biasConnect(3) = 1;
net.biasConnect(4) = 1;

%% define nodes and weights initial values
net.initFcn = 'initlay';
net.layers{1}.initFcn = 'initwb';
net.layers{2}.initFcn = 'initwb';
net.layers{3}.initFcn = 'initwb';
net.layers{4}.initFcn = 'initwb';
net.layers{5}.initFcn = 'initwb';
net.layerWeights{5,1}.initFcn = 'randsmall';
net.layerWeights{5,2}.initFcn = 'randsmall';
net.layerWeights{5,3}.initFcn = 'randsmall';
net.layerWeights{5,4}.initFcn = 'randsmall';

%% define layer functions
net.layers{1}.transferFcn = 'sintransfer';
net.layers{2}.transferFcn = 'purelin';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'poslin';
net.layers{5}.transferFcn = 'purelin';

%% define neural network parameters, partition data into train, validation, and test
net.divideFcn = 'divideind';  %Partition indices should be decided
net.divideParam.trainInd = 1:trainind;
net.divideParam.valInd = trainind+1:valind;
net.divideParam.testInd = valind+1:testind; 
net.divideMode = 'time';  % Divide up every sample
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};
net.trainFcn = 'trainlm';
net.performFcn = 'mae';
net.trainParam.lr = 1e-3;
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 1e-5;
% net.trainParam.min_grad = 1e-10;
net.trainParam.max_fail = 30;
net = configure(net, input, output);

%% initialize input weights and bias 
a = [];
b = [];
c = ones(net.layers{2}.size,1);
d = ones(net.layers{2}.size,1);
e = ones(net.layers{3}.size,1);
f = zeros(net.layers{3}.size,1);
g = ones(net.layers{4}.size,1);
h = zeros(net.layers{4}.size,1);

for k = 1:net.layers{1}.size
    a = [a, 2*pi*(k/2)];
    if mod(k,2)==0
        b = [b, pi/2];
    else
        b = [b, pi];
    end
end

net.IW{1,1} = a';
net.IW{2,1} = c;
net.IW{3,1} = e;
net.IW{4,1} = g;
net.b{1,1} = b';
net.b{2,1} = d;
net.b{3,1} = f;
net.b{4,1} = h;
% net.LW{5,1} = l;
% net.LW{5,2} = m;
% net.LW{5,3} = n;
% net.LW{5,4} = o;

%% define weights and biases learning functions
net.biases{1}.learnFcn = 'learngd';
net.biases{2}.learnFcn = 'learngd';
net.biases{3}.learnFcn = 'learngd';
net.inputWeights{1,1}.learnFcn = 'learngd';
net.inputWeights{2,1}.learnFcn = 'learngd';
net.inputWeights{3,1}.learnFcn = 'learngd';
net.layerWeights{5,1}.learnFcn = 'learngd';
net.layerWeights{5,2}.learnFcn = 'learngd';
net.layerWeights{5,3}.learnFcn = 'learngd';
net.layerWeights{5,4}.learnFcn = 'learngd';

%% train neural network and output
net.performFcn = 'mae';
[net,tr, ab ,er] = train(net, input, output);
perf = mae(er);
y = net(input);
result = fromnndata(y, wasMatriy, false, false);
Result = result.*max(prep(end-init+1:end)); 

% plot observed and simulated data
figure(3)
plot(time(end-init+1:end), prep(end-init+1:end), 'b-.')
hold on, grid on, grid minor
plot(time(end-init+1:end), Result, 'r')
xlim([time(end-init+1) time(end)])
title('Monthly total precipitation')
xlabel('year')
ylabel('mm')
legend('observed','simulated','location','best')
xlim([235 499])
xticks([235 247 259 271 283 295 307 319 331 343 355 367 379 391 403 415 427 439 451 463 475 487 499])
xticklabels({'Jul-1996','Jul-1997','Jul-1998','Jul-1999',...
    'Jul-2000','Jul-2001','Jul-2002','Jul-2003','Jul-2004','Jul-2005','Jul-2006','Jul-2007','Jul-2008','Jul-2009','Jul-2010',...
    'Jul-2011','Jul-2012','Jul-2013','Jul-2014','Jul-2015','Jul-2016','Jul-2017','Jul-2018'})
xtickangle(90)

figure, plotperform(tr)
