# collaborative_filtering
Two different approaches in Matlab for implementing a collaborative filtering system to predict users' ratings. This project is based on MovieLens-100K dataset. 

Files:

*main_v1.m* - Implementation of a collaborative filtering following the idea of closest neighbours. That is, first of all I make a prediction of clusters with closest neighbours after setting a correlation threshold. So clusters are composed by users with similar tastes. Afterwards, a matrix factorization algorithm is applied within every cluster. In particular, it is **SVD** algorithm for predicting users' ratings. 

*main_v2.m* - Implementation of a collaborative filtering WITHOUT clustering. After predicting users' ratings by means of a simple **SVD** algorithm, I apply an optimization method for the predictions based on a neural network. The neural network is a **Multilayer Perceptron (MLP)** of 4 layers traines over 100 epochs. In order to compose the input to the network, I concatenate the matrix with our previous predictions, information regarding the gender of users and genres of movies. The target is the matrix of real ratings. 

Folders:

*~/MLP* - Code implementing the artificial neural network.

*~/data_information* - Excel sheets with information about the gender of users and genre of movies. 

*~/data_utils* - Matlab data stored after several attempts. It enhances and speed up the process. 
