#include <iostream>
#include <mlpack/core.hpp>
#include <tuple>
#include "logistic_function.hpp"
using matrix=arma::Mat<double>;
size_t const max_epochs = 100; 
const double lambda = 0.5;
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

matrix EvaluateBias(matrix const& inp, matrix const& hiddenLayer, matrix const& outputLayer)
	{
	
	matrix hidden = (inp.t()*hiddenLayer).t(); // m*22*22*4 -> 4*m
	hidden = LogisticFunction::fn(hidden);//4*m
	matrix hiddenInput = arma::join_cols(arma::ones(1, inp.n_cols), hidden);//5*m
	matrix output = hiddenInput.t()*outputLayer; // m*5*5*3 -> m*3
	return LogisticFunction::fn(output.t());
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
	matrix const& X, matrix const& a1, matrix const& y, matrix const& Theta1, matrix const& Theta2
	)
	{
	matrix theta1_gradient(Theta1);
	theta1_gradient.zeros();
	matrix theta2_gradient(Theta2);
	theta2_gradient.zeros();
	size_t number_of_inputs = X.n_cols;
	matrix output(Theta2.n_cols, 1);
	output = EvaluateBias(a1, Theta1, Theta2);
	
	double cost = CalcCost(y, output);
	size_t m = X.n_cols;
	
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
	matrix const& X, matrix const& y,size_t hidden_layer_size
	)
	{
	matrix theta1;
	theta1.randu(X.n_rows+1, hidden_layer_size);
	matrix theta2;
	theta2.randu(hidden_layer_size+1, y.n_rows);
	double eta =0.5;
	size_t m = X.n_cols;
	matrix a1 = arma::join_cols(arma::ones(1, m), X); 
	for(size_t epoch = 0; epoch < max_epochs; ++epoch)
		{
		matrix delta1,delta2;
		double cost=0.0;
		std::tie(cost,delta1,delta2) = BackProp(X,a1,y,theta1,theta2);
		theta1 = theta1 - eta *  delta1;
		theta2 = theta2 - eta *  delta2;
		std::cout << cost << std::endl;
		}
	return std::make_tuple(theta1,theta2); 
	}

void main()
	{
	// Load the dataset.
	using namespace mlpack;
	arma::mat dataset;
	data::Load("thyroid_train.csv", dataset, true);

	arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
		dataset.n_cols - 1);
	arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0,
		dataset.n_rows - 1, dataset.n_cols - 1);
	matrix theta1, theta2;
	std::tie(theta1, theta2) = TrainNetwork(trainData, trainLabels, 4);

	data::Load("thyroid_test.csv", dataset, true);

	arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
		dataset.n_cols - 1);
	arma::mat testLabels = dataset.submat(dataset.n_rows - 3, 0,
		dataset.n_rows - 1, dataset.n_cols - 1);
	}