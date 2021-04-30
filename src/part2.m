load AC50001_assignment2_data.mat %load the data set


%class means
mu8=mean(digit_eight');
mu5=mean(digit_five');
mu1=mean(digit_one');

%covariance mat
s8=cov(digit_eight');
s5=cov(digit_five');
s1=cov(digit_one');

%scatter within classes
sw=s8+s5+s1;

mu=(mu8+mu5+mu1)/3;

%Between means cov
meanCov8=(mu8-mu)'*(mu8-mu);
meanCov5=(mu5-mu)'*(mu5-mu);
meanCov1=(mu1-mu)'*(mu1-mu);

%scatter between classes
sb=(meanCov8+meanCov5+meanCov1);
lda=pinv(sw)*sb;

[V,D]=eig(lda);

 %Sort the eigvalues
 eigValues=diag(D);
 [dSorted,sortedIdx]=sort(eigValues,'descend');
 V=V(:,sortedIdx);
    
 new8=digit_eight'*V(:,1:2);
 new5=digit_five'*V(:,1:2);
 new1=digit_one'*V(:,1:2);
 datapoints=[new8;new5;new1];
 classlabel=[repmat(8,100,1);repmat(5,100,1);ones(100,1)];

 rng(1)%to get the same result    
%performing classification using kmeans
[idx,C] = kmeans(datapoints,3);
x1 = min(datapoints(:,1)):0.01:max(datapoints(:,1));
x2 = min(datapoints(:,2)):0.01:max(datapoints(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
idx2Region = kmeans(XGrid,3,'MaxIter',1,'Start',C);

%plotting datapoints 
figure;
hold on
title('DATA POINTS IN 2PCs SPACE')
xlabel('W 1')
ylabel('W 2')
gscatter(datapoints(:,1),datapoints(:,2),classlabel,'rgb','osd');
legend('8','5','1','location','SE');
hold off;


%plotting clustered dataset
figure;
hold on
title('TWO PCs REPRESENTATION OF IMAGES')
xlabel('W 1')
ylabel('W 2')
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    'ycg','..');
gscatter(datapoints(:,1),datapoints(:,2),classlabel,'rbk','.*+');
legend('C 1','C 2','C 3','8','5','1','location','SE');
hold off;

%let row 1,2,3 stand for digit 8,5,1 
classCluster(1,:)=sum(idx(1:100)==[1,2,3]);
classCluster(2,:)=sum(idx(101:200)==[1,2,3]);
classCluster(3,:)=sum(idx(201:300)==[1,2,3]);

disp("let row 1,2,3 be digit 8,5,1; and col 1,2,3 be cluster 1,2,3");
disp("CONFUSION MATRIX");
disp(classCluster)

%clear;

