load AC50001_assignment2_data.mat;

%create the two classes, 
%let class 1 be digit 5 and class 0 be digit 1 and 8
class_label=[ones(100,1);zeros(200,1)];

%merge datasets into one and transpose them with features pixels as columns
data=[digit_five';digit_one';digit_eight'];

rng(1)%for the same result
%set up 5 fold cross validation partitons
k=5;
cvo = cvpartition(class_label,'k',k);

%Create network with 1 hidden layer
net = patternnet(1); 

%initialize probaility matrix for use in roc
NNProbability=zeros(300,2);
confusionMat=zeros(2,2);
row=1;
for i=1:k
    trIdx = cvo.training(i); % get index of training samples
    teIdx = cvo.test(i); % get the index of test samples
    
    training_label_vector = class_label(trIdx); % training set labels
    training_instance_matrix = data(trIdx,:); % training set feature vectors
    
    test_label_vector = class_label(teIdx); % test set labels
    test_instance_matrix = data(teIdx,:);
    
    %train network
    net = train(net,training_instance_matrix',training_label_vector');
    
    %perform testing
    y = net(test_instance_matrix');
    
    %confusion matrix
    confusionMat=confusionMat+confusionmat(logical(test_label_vector),any(y'>=0.5,2));
    
    %creating data mat for ROC curve
    NNProbability(row:numel(test_label_vector)*i,1)=test_label_vector;
    NNProbability(row:numel(test_label_vector)*i,2)=y;
    
    row=(numel(test_label_vector)*i)+1;
end
classyAccurracy=(confusionMat(1,1)+confusionMat(2,2))/300;

disp("Confusion matrix=");
disp(confusionMat);
disp("Classification Accuracy=");
disp(classyAccurracy);  

%save data to LKProbability.mat
save NNProbability.mat NNProbability;

clear;