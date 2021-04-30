load AC50001_assignment2_data.mat %load the data set

%create the two classes, 
%let class 1 be digit 5 and class 0 be digit 1 and 8
class_label=[ones(100,1);zeros(200,1)];

%merge datasets into one and transpose them with features pixels as columns
data=[digit_five';digit_one';digit_eight'];

rng(1)%for the same result
%set up 5 fold cross validation partitons
k=5;
cvo = cvpartition(class_label,'k',k);

%initialize probaility matrix for use in roc
RBFProbability=zeros(300,2);
confusionMat=zeros(2,2);
row=1;
for i=1:k
    trIdx = cvo.training(i); % get index of training samples
    teIdx = cvo.test(i); % get the index of test samples
    
    training_label_vector = class_label(trIdx); % training set labels
    training_instance_matrix = data(trIdx,:); % training set feature vectors
    
    test_label_vector = class_label(teIdx); % test set labels
    test_instance_matrix = data(teIdx,:);% test set feature vectors
    %train model
    model = svmtrain(training_label_vector,training_instance_matrix, '-t 2 -g 0.01 -c 1000 -b 1 -q');

    %apply model on test set for prediction
    [predict_label, accuracy, dec_values] = svmpredict(test_label_vector,test_instance_matrix, model,'-b 1 -q');
    
    %confusion matrix
    confusionMat=confusionMat+confusionmat(test_label_vector,predict_label);
    
    %creating data mat for ROC curve
    RBFProbability(row:numel(test_label_vector)*i,1)=test_label_vector;
    RBFProbability(row:numel(test_label_vector)*i,2)=dec_values(1:numel(test_label_vector),1);
    
    row=(numel(test_label_vector)*i)+1;

end  

classyAccurracy=(confusionMat(1,1)+confusionMat(2,2))/300;

disp("Confusion matrix=");
disp(confusionMat);
disp("Classification Accuracy=");
disp(classyAccurracy);

%save data to LKProbability.mat
save RBFProbability.mat RBFProbability;

clear;


