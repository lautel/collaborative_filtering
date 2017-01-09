%% DataBase: MovieLens 100K (shortened)
% Author: Laura Cabello Piqueras

% VERSION 1.0
% Implmentation of a collaborative filtering following the idea of closest
% neighbours. That is, first of all I make a prediction of clusters with 
% closest neighbours after setting a correlation threshold. So clusters are
% composed by users with similar tastes. 
% Afterwards, a matrix factorization algorithm is applied within 
% every cluster. In particular, it is SVD algorithm for predicting users' 
% ratings. 
% Finally the rmse rate (as did in Netflix competition) is evaluated with a
% different test set. 

% CONCLUSIONS: this method is not worth it!! It requires too much 
% computational operations and the results are not good enough. So that's
% why in main_v2.m I redo the same experiment although skipping the
% clustering task. So the prediction is based upon the total group of
% ratings and users. 


clear all
close all
clc

%% PRE-PROCESSING OF DATA

% Train = xlsread('train1.xlsx');
% Test = xlsread('test1.xlsx');

load data_utils/Train.mat
load data_utils/Test.mat

% Train 
% Working with a 400 users subset for higher speed in initial tests
u400 = find(Train(:,1)==400);
Tr_user = Train(1:u400(end),1); 
Tr_item = Train(1:u400(end),2);
Tr_rating = Train(1:u400(end),3);

% Tr_user = Train(:,1); 
% Tr_item = Train(:,2);
% Tr_rating = Train(:,3);

% Test
% Working with a 400 users subset for higher speed in initial tests
u400 = find(Test(:,1)==400);
Ts_user = Test(1:u400(end),1);
Ts_item = Test(1:u400(end),2);
Ts_rating = Test(1:u400(end),3);

% Ts_user = Test(:,1);
% Ts_item = Test(:,2);
% Ts_rating = Test(:,3);

% CELL ARRAY of users: 'user','item','rating'
n_items = max(Tr_item);
n_users = max(Tr_user); 
users_data = cell(n_users,1);

% Matrix of ratings: users along columns and movies along rows
% 'riu' matrix for train and 'R_test' matrix for test
riu = NaN(n_items,n_users);
R_test = NaN(n_items,n_users);

for k = 1:n_users
    
    % Train 
    movie = Tr_item(Tr_user == k);
    riu(movie,k) = Tr_rating(find(Tr_user == k));
    
    us = find(Tr_user==k);
    movie_user = Tr_item(us);
    rating_user = Tr_rating(us);
    
    users_data{k}(:,1) = movie_user;
    users_data{k}(:,2) = rating_user;
    
    % Test
    movie2 = Ts_item(Ts_user == k);
    R_test(movie2,k) = Ts_rating(find(Ts_user == k));
end

%% COLLABORATIVE FILTERING

% Check whether every movie has been rated by at least 1 user or not. 
if_ratings = zeros(1,n_items);
for k = 1:n_items
    if_ratings(k) = length(find(isnan(riu(k,:))==0));
end
figure, stem(if_ratings)
title('Number of ratings per movie')

% In the event that a movie hasn't been rated, it is deleted.
riu = riu(1:length(find(if_ratings>0)),:);
n_items = size(riu,1);

%% PEARSON'S CORRELATION: CLOSEST NEIGHBOURS

mu_user = nanmean(riu,1);           % Average rating per user
mu_track = nanmean(riu,2);          % Average rating per movie

riu = riu - repmat(mu_user,size(riu,1),1);

sims = corr(riu, 'rows', 'pairwise'); 
sims = sims - eye(n_users);         % Set autocorrelation to 0

% Initialize cell arrays
ngh_corr_true = cell(1,size(sims,2));
ngh_idx_true = cell(1,size(sims,2));
ngh_mu = cell(1,size(sims,2));
ngh_r = cell(1,size(sims,2));


for k = 1:n_users
    user_corr = sims(:,k);
    [ngh_corr, ngh_idx] = sort(user_corr,'descend');
    
    non_ngh = find(ngh_corr==0);    % If correlación<=0 --> No neighbour
    ngh_corr(non_ngh:end) = [];     % Those who aren't neigh are eliminated
    ngh_idx(non_ngh:end) = [];      % Those who aren't neigh are eliminated
    
    % I keep real neighbours and delete NaNs
    isNaN = ~isnan(ngh_corr); 
    ngh_corr_true{k} = ngh_corr(isNaN);
    ngh_idx_true{k} = ngh_idx(isNaN);
    
    % Average rating from my neighbours
    ngh_mu{k} = nanmean(riu(:,ngh_idx_true{k}),1);    
 
    clear ngh_corr ngh_idx
end

%% Clustering neighbours

us = ones(n_users,1);
clust = 1;
umbral = 0.5; % Decision threshold

for iuser = 1:n_users
    
    if us(iuser) == 1
        
        aux = find(ngh_corr_true{iuser} > umbral);
        
        if ~isempty(aux)
            aux2 = [iuser; ngh_idx_true{iuser}(aux)];
            
            cluster{clust} = sort(aux2,'ascend');  
            mu_cluster(clust) = nanmean(mu_user(cluster{clust}));    

            us([iuser;cluster{clust}]) = 0;
            clust = clust+1;
        end       
    end   
end

%% Check the movies that have been rated within some sample clusters

clustern = randi(size(cluster,2),1,10);
figure,
for c = 1:10
    users_c = cluster{1,clustern(c)};
    subplot(10,1,c)
    for cc = 1:length(users_c)
        z = users_data{users_c(cc),1};
        z = z(:,1);
        stem(z,1:length(z))
        xlim([1,1690])      % 1682 movies
        hold on
    end
end


% Here we find the main pitfall of this method: there are several movies
% that are not rated within a cluster, so its prediction will be 'blind',
% leading to a high error rate.

%% Hiperparameters tuning with a separate subset of data (DEV)

% [alpha, beta, K] = dev_ajustes(riu, cluster, R_test)
alpha = 0.0007;
beta = 0.02;
K=10;

%% SVD in clusters

N = size(riu,1);
M = size(riu,2);
[Pe, Qe] = my_svd(riu, cluster, N, M, K, alpha, beta);


% Calculate RMSE y and standart deviation

tabla = NaN(10,n_users); % There are 10 items per user in test set
for j = 1:size(cluster,2)  % Clusters
    
    R_est = round(Pe{1,j}*Qe{1,j});
    users_en_cluster = cluster{1,j};
    
    for i = 1:length(users_en_cluster)
                    
            local = find(isnan(R_test(:,users_en_cluster(i))) == 0);
            rating_test = R_test(local, users_en_cluster(i));
            rating_estimado = R_est(local,users_en_cluster(i));
            
            rmse(:,users_en_cluster(i)) = abs(sqrt(rating_estimado.^2 - rating_test.^2));
            
            tabla(:,users_en_cluster(i)) = rating_estimado; 
    end
    
end


% Plotting results...

figure,plot(mean(rmse,1),'*')
xlabel('Users')
ylabel('RMSE')




