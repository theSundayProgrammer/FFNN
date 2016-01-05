#pragma once
typedef arma::Mat<double> matrix;

/**
Evaluate Output
Given
Input  data X(input_dim,1)
hidden-layer (input_dim,hidden_layer_size)
Output (hidden_layer_size,out_put_size)
Compute
Output of FFNN

*/
matrix
Evaluate(matrix const& X, matrix const& hiddenLayer, matrix const& outputLayer);

/**
Compute Output including bias
Given
Input  data X(input_dim,1)
hidden-layer (input_dim,hidden_layer_size)
Output (hidden_layer_size,out_put_size)
Compute
Output of FFNN

*/
matrix
EvaluateWithBias(matrix const& inp, matrix const& hiddenLayer, matrix const& outputLayer);

/**
Calculate cost using cross entropy
*/
double
CalcCost(matrix const& target, matrix const& output);



/**Back Propogation
Ignore lambda
Problem statement:
Given
X the feature mector of m column vectors of size feature_size
Y the result vector of m column vectors of label_size
hidden_layer_size
Theta1: weight vector of hidden_layer_size columns each of size (feature_size )
Theta2: weight vector of label_size columns each of size (hidden_layer_size)
Compute
cost using cross entropy
Theta1_gardient: weight vector of hidden_layer_size columns each of size (feature_size )
Theta2_gradient: weight vector of size label_size columns each of size (hidden_layer_size )
...using sigmoid

Updates:
Adding bias

*/
std::tuple<double, matrix, matrix>
BackProp(
	matrix & a1, matrix & y, matrix const& Theta1, matrix const& Theta2
	);


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