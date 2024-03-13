data=xlsread('C:\云盘\testoptimized_parameters.xls');

inputData1 = data(:, 1:2);
targetData1 = data(:, 3:end); 

inputData= inputData1;
targetData = targetData1;

hiddenLayerSize = 20;
net = feedforwardnet(hiddenLayerSize);

net.trainFcn = 'traingd'; 

net.performFcn = 'mse'; 


net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


net.trainParam.epochs = 200; 
net.trainParam.lr = 0.01;    
net.trainParam.max_fail = 6;   
net.trainParam.min_grad = 1e-7;


[net,tr] = train(net,inputData',targetData');


figure;
plot(tr.epoch, tr.perf, 'b');
hold on;
plot(tr.epoch, tr.vperf, 'r');
plot(tr.epoch, tr.tperf, 'g');
title('Training, Validation and Test Performance');
xlabel('Epochs');
ylabel('Performance');
legend('Training', 'Validation', 'Test');


figure(20);
plot(tr.epoch, tr.perf, 'b','LineWidth',1.1);
hold on;
plot(tr.epoch, tr.vperf, 'r','LineWidth',1.1);
plot(tr.epoch, tr.tperf, 'g','LineWidth',1.1);
ylim([0,1000 ...
    ]);





outputs = net(inputData');



outputs = net(inputData');
errors = gsubtract(targetData', outputs);
performance = perform(net,targetData', outputs);


view(net);
figure, plotperform(tr);
figure, plottrainstate(tr);
figure, plotregression(targetData', outputs);
figure, ploterrhist(errors);


trainPerformance = tr.best_perf;
valPerformance = tr.best_vperf;
testPerformance = tr.best_tperf;

disp(['Training Performance: ' num2str(trainPerformance)])
disp(['Validation Performance: ' num2str(valPerformance)])
disp(['Test Performance: ' num2str(testPerformance)])


figure, plot(tr.epoch, tr.perf, 'b', tr.epoch, tr.vperf, 'r', tr.epoch, tr.tperf, 'g')
title('Training, Validation and Test Performance');
xlabel('Epochs');
ylabel('Performance');
legend('Training', 'Validation', 'Test');




numSamples = 40;
indices = randperm(size(inputData1, 1), numSamples);

selectedTargetData = targetData1(indices, :); 
selectedInputData = inputData1(indices, :); %
predictedData = net(selectedInputData'); 


accuracy = perform(net, selectedTargetData', predictedData);

% 显示预测精度
disp(['Prediction Accuracy (MSE): ' num2str(accuracy)]);
