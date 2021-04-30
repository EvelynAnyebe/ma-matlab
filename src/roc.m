function [fpr,tpr, auc]=roc(data)

%this function returns roc values and auc for a classifier
threshold=sort(data(:,2),1,'descend')';
val=numel(threshold);
tpr=ones(val,1);
fpr=ones(val,1);
for i=1:val
   
    %get all class 1 values as positives
    p=data(data(:,1)==1,:);
    
    %get all class 2 values as negatives
    n=data(data(:,1)==0,:);
    
    %create logical vector of values for each class based on threshold
    classifyp=p(:,2)>=threshold(i);
    classifyn=n(:,2)>=threshold(i);
    
    %get true positive
    tp=p(classifyp);
    
    %get false negative
    fn=p(~classifyp);
    
    %get false positive
    fp=n(classifyn);
    
    % get true positive
    tn=n(~classifyn);
    
    %get true positive rate and false positive rate for every threshold
    tpr(i)=numel(tp)/(numel(tp)+numel(fn));
    fpr(i)=numel(fp)/(numel(fp)+numel(tn));
end


%auc= sum of area of each trapeziod in curve.
auc=trapz(fpr,tpr);


