#include <iostream>
#include <armadillo>
#include <tuple>
#include <cstdlib>     /* srand, rand */
#include <ctime>       /* time */
#include "logistic_function.hpp"
#include "neural.hpp"
using matrix=arma::Mat<double>;
size_t const max_epochs = 1000; 
const double lambda = 0.8;
const double tolerance = 0.000001;
/**
Evaluate Output
 Given
  Input  data X(input_dim,1)
  hidden-layer (input_dim,hidden_layer_size)
  Output (hidden_layer_size,out_put_size)
 Compute 
  Output of FFNN
   
*/
matrix Evaluate(matrix const& X, matrix const& hiddenLayer, matrix const& outputLayer)
	{
	matrix inp = X; //21*m
	matrix hidden = inp.t()*hiddenLayer; // m*21*21*4 -> m*4
	matrix output = LogisticFunction::fn(hidden)*outputLayer; // m*4 * 4 *3 -> m*3
	return LogisticFunction::fn(output).t();//3*m
	}

matrix EvaluateWithBias(matrix const& inp, matrix const& hiddenLayer, matrix const& outputLayer)
	{
	
	matrix hidden = hiddenLayer.t()*inp; //  4*m
	hidden = LogisticFunction::fn(hidden);//4*m
	matrix hiddenInput = arma::join_cols(arma::ones(1, inp.n_cols), hidden);//5*m
	matrix output = outputLayer.t()*hiddenInput; // (5.3)'*5.m -> 3.m
	return LogisticFunction::fn(output);
	}
/**
Calculate cost using cross entropy
*/
double CalcCost(matrix const& target, matrix const& output)
	{

	double sum = 0.0;
	size_t item_count = target.n_elem;
	for (size_t i = 0; i < item_count; ++i)
		{
		double y = target[i];
		double val = output[i];
		sum = sum - y*log(val) - (1 - y)*log(1 - val);
		}
	return sum / target.n_cols;
	}


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
std::tuple<double,matrix,matrix> 
BackProp(
	 matrix & a1, matrix & y, matrix const& Theta1, matrix const& Theta2
	)
	{

	// This generates [0 1 2 3 ... (ElementCount(trainingData) - 1)]. The
	// sequence will be used to iterate through the training data.
	size_t m = a1.n_cols;


	matrix theta1_gradient(Theta1);
	theta1_gradient.zeros();
	matrix theta2_gradient(Theta2);
	theta2_gradient.zeros();
	
	matrix output(Theta2.n_cols, 1);

	for (size_t j = m; j > 0; --j)
		{
		int k = rand() % j;
		a1.swap_cols(j - 1, k);
		y.swap_cols(j - 1, k);
		}
	output = EvaluateWithBias(a1, Theta1, Theta2);
	double cost = CalcCost(y, output);
	
	matrix z2 = Theta1.t()*a1;
	matrix a2 = arma::join_cols(arma::ones(1, m), LogisticFunction::fn(z2));

	matrix z3 = Theta2.t()*a2;
	matrix a3 = LogisticFunction::fn(z3);
	matrix delta3 = a3 - y;
	theta2_gradient += a2*delta3.t();
	matrix delta2 = (Theta2*delta3) % arma::join_cols(arma::ones(1, m), LogisticFunction::deriv(a2.rows(1, a2.n_rows - 1)));
	delta2 = delta2.rows(1, delta2.n_rows-1);
	theta1_gradient += a1*delta2.t();
	theta1_gradient = theta1_gradient / m + lambda*arma::join_cols(arma::zeros(1, Theta1.n_cols), Theta1.rows(1, Theta1.n_rows - 1));
	theta2_gradient = theta2_gradient / m + lambda*arma::join_cols(arma::zeros(1, Theta2.n_cols), Theta2.rows(1, Theta2.n_rows - 1));
	return std::make_tuple(cost, theta1_gradient, theta2_gradient);
	}

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

std::tuple<matrix,matrix>
TrainNetwork(
	matrix const& X, matrix y,size_t hidden_layer_size
	)
	{
	matrix theta1;
	theta1.randu(X.n_rows+1, hidden_layer_size);
	matrix theta2;
	theta2.randu(hidden_layer_size+1, y.n_rows);
	double eta =0.5;
	size_t m = X.n_cols;
	matrix a1 = arma::join_cols(arma::ones(1, m), X); 
	double prev_cost = 1000.0;
	for(size_t epoch = 0; epoch < max_epochs; ++epoch)
		{
		matrix delta1,delta2;
		double cost=0.0;
		std::tie(cost,delta1,delta2) = BackProp(a1,y,theta1,theta2);
		theta1 = theta1 - eta *  delta1;
		theta2 = theta2 - eta *  delta2;
		//std::cout << epoch << ":" << cost << std::endl;
		if (abs(prev_cost - cost) < tolerance)
			break;
		prev_cost = cost;
		}
	return std::make_tuple(theta1,theta2); 
	}

/**
Predict 
	Given 
	 Neural Network
	 Input sample
    Compute 
	 Class it belongs to
*/
matrix Predict(matrix const& input, matrix const& hiddenLayer, matrix const& outputLayer)
	{
	matrix output = EvaluateWithBias(input, hiddenLayer, outputLayer);
	matrix result(output.n_rows, output.n_cols);
	for (size_t i = 0; i < output.n_cols; ++i)
		{
		matrix cur = output.unsafe_col(i);
		double max = cur[0];
		size_t index = 0;
		for (size_t j = 1; j < cur.n_elem; ++j)
			{
			if (cur[j] > max)
				{
				max = cur[j];
				index = j;
				}
			}
		for (size_t j = 0; j < output.n_rows; ++j)
			{
			result(j, i) = index == j ? 1.0 : 0.0;
			}
		}
	return result;
	}
double ComputeError(matrix const& result, matrix const& labels)
	{
	double error = 0.0;
	for (size_t j = 0; j < result.n_elem; ++j)
		error += abs(result[j] - labels[j]);
	return error / 2.0;
	}
