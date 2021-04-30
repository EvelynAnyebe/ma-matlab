%threshold=0:0.005:1;
figure;
title("ROC CURVE OF 3 CLASSIFIERS");
hold on;

%ploting for linear kernel
load('LKProbability.mat');
[x,y,LKAUC]=roc(LKProbability);
plot(x,y,'r','LineWidth',1);


%ploting for RBF classifier
load('RBFProbability.mat');
[x,y,RBFAUC]=roc(RBFProbability);
plot(x,y,'b','LineWidth',1);

%ploting for NN classifier
load('NNProbability.mat');
[x,y,NNAUC]=roc(NNProbability);
plot(x,y,'g','LineWidth',1);


xlabel('FPR');
ylabel('TPR');
xlim([0,1]);
ylim([0,1]);
legend('Linear Kernel','RBF','Neural Network','location','SE');

hold off;