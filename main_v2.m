%% DataBase: MovieLens 100K (shortened)
% Author: Laura Cabello Piqueras

% VERSION 2.0
% Implementation of a collaborative filtering over the entire training 
% dataset. A matrix factorization algorithm is applied. In particular, it 
% is SVD algorithm for predicting users' ratings.
% Finally the rmse rate (as did in Netflix competition) is evaluated with 
% a different test set. 

% OPTIMIZATION: The last step in this recommendation system is to apply an
% optimization technique to those ratings predicted. I implement a neural
% network taking into account information such as the gender of the user or
% the genres of the movies. In such a way, the incoming to the network is a
% vector with the exact ratings (removed from the training set) and a
% matrix composed by our prediction and the extra information we got. 

% CONCLUSIONS: On average, the final rmse is lower than the calculated with
% the clustering technique. In addition, it is successfully lessen with
% the optimization process after few iterations. 

% PRO: this technique is computationally far more efficient than the 
% previous one.

clear all
close all
clc

%% DATA PRE-PROCESSING

% Train = xlsread('train1.xlsx');
% Test = xlsread('test1.xlsx');

load data_utils/Train.mat
load data_utils/Test.mat

% Train 
% Working with a 400 users subset for higher speed in initial tests
% u400 = find(Train(:,1)==400);
% Tr_user = Train(1:u400(end),1); 
% Tr_item = Train(1:u400(end),2);
% Tr_rating = Train(1:u400(end),3);

Tr_user = Train(:,1); 
Tr_item = Train(:,2);
Tr_rating = Train(:,3);

% Test
% Working with a 400 users subset for higher speed in initial tests
% u400 = find(Test(:,1)==400);
% Ts_user = Test(1:u400(end),1);
% Ts_item = Test(1:u400(end),2);
% Ts_rating = Test(1:u400(end),3);

Ts_user = Test(:,1);
Ts_item = Test(:,2);
Ts_rating = Test(:,3);

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

% R matrix has a 1 if the user j has rated the movie i in the train set.
% It's zero otherwise. 
R = ones(size(riu));
R(isnan(riu)) = 0;

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


%% SVD 

% Hiperparameters tuning with a separate subset of data (DEV)

% [alpha, beta, K] = dev_ajustes(riu, cluster, R_test)

alpha = 0.0007;
beta = 0.02;
K = 10;

% Due to its strong effect over the final result, I saved the best 
% initialization for the matrices P and Q after several trials. This is
% highly recommended with a low amount of iterations. 
% Let's load them. 

% Pini = rand(n_items,K);    % items x K
% Qini = rand(n_users,K);    % users x K 

load data_utils/Pini.mat
load data_utils/Qini.mat

Pini = Pini(1:n_items,:);
Qini = Qini(1:n_users,:);

[Pe1, Qe1] = my_svd_uno(Pini, Qini, riu, R, n_items, n_users, K, alpha, beta);

R_est1 = round(Pe1*(Qe1.'));


% Calculating RMSE y and standart deviation

tabla = NaN(10,n_users); % There are 10 items per user in test set
rmse = NaN(10,n_users);
users_en = 1:n_users;

for i = 1:length(users_en)
    local = find(isnan(R_test(:,users_en(i))) == 0);
    rating_test = R_test(local, users_en(i));    
    rating_estimado = R_est1(local,users_en(i));
            
    rmse(:,users_en(i)) = abs(sqrt(rating_estimado.^2 - rating_test.^2));            
    tabla(:,users_en(i)) = rating_estimado;      
end

% Plotting results...

% figure,plot(mean(rmse,1),'*')
% hold on, plot(1:n_users,mean(mean(rmse,1)),'r-')
% title('Mean of the RMSE of the predictions made per user (test)')
% legend('RMSE test','RMSE')
% xlabel('Users')
% ylabel('RMSE')

%% Optimization: a neural network

addpath(genpath('./test'));
addpath(genpath('./MLP'));

% Preparing data:

% Target data
Y = tabla(:);
Y=Y';
Y=Y./5;

% Predictions from the test set
x0 = R_test;
x0(isnan(x0))=[];
x0=x0./5;

% Extra information: gender and genres
descript = xlsread('data_information/genres.xlsx');
userInfo = xlsread('data_information/user.xlsx');

Genero = descript(:,3:end);
Genero = Genero';
U = userInfo(:,3);
U=U';

% Gender of users (0-Male, 1-Female)
x1 = zeros(1,10);
for i=2:length(U)
    x1 = [x1 repmat(U(i),1,10)];
end

% Genres of movies (19 genres)        
for i=1:length(Ts_item)
    for u=1:length(U)
        mov = Ts_item(i);
        x2(:,i) = Genero(:,mov);
    
    end
end

X = [x0;x1;x2];

B = 100;

mlp = initiate_MLP([size(X,1) 30 30 size(Y,1)], B);
mlp.learnrate = 0.10;

[mlp,mse_r] = train_MLP(mlp, X, Y, 100);





