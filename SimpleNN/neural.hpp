#pragma once
typedef arma::Mat<double> matrix;

/**
Train Network
Ignore lambda and bias
Given:
X the feature mector of m column vectors of size feature_size
Y the result vector of m column vectors of label_size
hidden_layer_size
Compute:
Theta1: weight vector of hidden_layer_size columns each of size (feature_size + 1)
Theta2: weight vector of size label_size columns each of size (hidden_layer_size +1)
Update:
Adding bias
*/

std::tuple<matrix, matrix>
TrainNetwork(
	matrix const& X, matrix y, size_t hidden_layer_size
	);


/**
Predict
Given
Neural Network
Input sample
Compute
Class it belongs to
*/
matrix Predict(matrix const& input, matrix const& hiddenLayer, matrix const& outputLayer);

double ComputeError(matrix const& result, matrix const& labels);