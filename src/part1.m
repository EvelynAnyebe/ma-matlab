load AC50001_assignment2_data.mat %load the data set

%im = reshape(digit_five(:,1), [28, 28]); % 1st image of digit ‘5’
%imshow(im,[]);

%concatenate all three digits to enable use with a for loop.
digits=[digit_eight';digit_five';digit_one'];

%Final data points initialized to stop matlab warning of varaible size changing.
datapoints=ones(300,2);

%indexing variable
startcol=1;
for i=1:3
    dataset=digits(startcol:i*100,:);
    
    %Translate data set
    X=dataset-mean(dataset);
    
    % Calculate eigenvalues and eigenvectors of the covariance matrix
    covarianceMatrix = X'*X;
    [V,D] = eig(covarianceMatrix);
    
    %Sort the eigvalues
    eigValues=diag(D);
    [dSorted,sortedIdx]=sort(eigValues,'descend');
    V=V(:,sortedIdx);
    
    %project data to 2-dim pc space to get data points
    datapoints(startcol:i*100,:)=X*V(:,1:2);
    
    startcol=(i*100)+1;
end    
classlabel=[repmat(8,100,1);repmat(5,100,1);ones(100,1)];


rng(1)%to get the same result
%performing classification using kmeans
opts = statset('Display','final');
%Distance 'cosine' produced a lower sum of distances
[idx,C] = kmeans(datapoints,3,'Distance','cityblock','Replicates',3,'Options',opts);
x1 = min(datapoints(:,1)):0.01:max(datapoints(:,1));
x2 = min(datapoints(:,2)):0.01:max(datapoints(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
idx2Region = kmeans(XGrid,3,'Maxiter',1,'Start',C);

%plotting datapoints 
figure;
subplot(1,2,1);
hold on
title('DATA POINTS IN 2PCs SPACE')
xlabel('PC 1')
ylabel('PC 2')
gscatter(datapoints(:,1),datapoints(:,2),classlabel,'rgb','.*+');
legend('8','5','1','location','SE');
hold off;

subplot(1,2,2);
hold on
title('CLUSTER OF DATA POINTS')
xlabel('PC 1')
ylabel('PC 2')
gscatter(datapoints(:,1),datapoints(:,2),idx,'ycg','.');
legend('Cluster 1','Cluster 2','Cluster 3','location','SE');
hold off;

%plotting cluster regions and data points 
figure;
hold on
title('KMEANS CLUSTERING OF DATA POINTS IN 2PCs SPACE')
xlabel('PC 1')
ylabel('PC 2')
%Drawing the region of each cluster
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    'ycg','..');
%ploting data points to show the cluster each has been assigned
gscatter(datapoints(:,1),datapoints(:,2),classlabel,'rbk','.*+');
legend('Cluster 1','Cluster 2','Cluster 3','8','5','1','location','SE');
hold off;

%let row 1,2,3 stand for digit 8,5,1 
classCluster(1,:)=sum(idx(1:100)==[1,2,3]);
classCluster(2,:)=sum(idx(101:200)==[1,2,3]);
classCluster(3,:)=sum(idx(201:300)==[1,2,3]);

disp("let row 1,2,3 be digit 8,5,1; and col 1,2,3 be cluster 1,2,3");
disp("CONFUSION MATRIX");
disp(classCluster)

clear digits i startcol X V sortedIdx D eigValues dSorted dataset covarianceMatrix;