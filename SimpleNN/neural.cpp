#include <iostream>
#include <mlpack/core.hpp>
#include <tuple>
#include "logistic_function.hpp"
using matrix=arma::Mat<double>;
/*
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
	return LogisticFunction::fn(LogisticFunction::fn(X.t()*hiddenLayer)*outputLayer).t();
	}


/*Back Propogation
 Ignore lambda and bias
 Problem statement:
 Given 
  X the feature mector of m column vectors of size feature_size
  Y the result vector of m column vectors of label_size
  hidden_layer_size 
  Theta1: weight vector of hidden_layer_size columns each of size (feature_size )
  Theta2: weight vector of label_size columns each of size (hidden_layer_size)
Compute
  cost
  Theta1_gardient: weight vector of hidden_layer_size columns each of size (feature_size )
  Theta2_gradient: weight vector of size label_size columns each of size (hidden_layer_size )
*/
double CalcCost(matrix const& target, matrix const& output)
	{
	
	double sum = 0.0;
	size_t item_count = target.n_cols;
	for (size_t i = 0; i < item_count; ++i)
		{
		auto y = target.col(i);
		for (size_t j = 0; j < y.n_elem; ++j)
			{
			sum = sum - y(j)*log(output(j, i)) - (1 - y(j))*log(1 - output(j,i));
			}
		}
	return sum/target.n_elem;
	}
std::tuple<double,matrix,matrix> 
BackProp(
	matrix const& X, matrix const& y, matrix const& Theta1, matrix const& Theta2
	)
	{
	matrix theta1_gradient(Theta1);
	theta1_gradient.zeros();
	matrix theta2_gradient(Theta2);
	theta2_gradient.zeros();
	size_t number_of_inputs = X.n_cols;
	matrix output(Theta2.n_cols, 1);
	output = Evaluate(X, Theta1, Theta2);
	
	double cost = CalcCost(y, output);
	size_t m = X.n_cols;
	for (size_t i = 0; i < m ; ++i)
		{
		matrix x = X.col(i);
		matrix z1 = Theta1.t()*x; 
		matrix a1 = LogisticFunction::fn(z1);
		matrix z2 = Theta2.t()*a1;
		matrix a2 = LogisticFunction::fn(z2);
		matrix delta3 = a2 - y.col(i);
		matrix delta2 = (Theta2*delta3) % LogisticFunction::deriv(a1);
		theta1_gradient += x*delta2.t();
		theta2_gradient += a1*delta3.t();
		}
	return std::make_tuple(cost, theta1_gradient/m, theta2_gradient/m);
	}

/*Train Network
 Ignore lambda and bias
 Given:
  X the feature mector of m column vectors of size feature_size
  Y the result vector of m column vectors of label_size
  hidden_layer_size 
 Compute:
  Theta1: weight vector of hidden_layer_size columns each of size (feature_size + 1)
  Theta2: weight vector of size label_size columns each of size (hidden_layer_size +1)
*/
size_t const max_epochs = 1; 
std::tuple<matrix,matrix>
TrainNetwork(
	matrix const& X, matrix const& y,size_t hidden_layer_size
	)
	{
	matrix theta1;
	theta1.randu(X.n_rows, hidden_layer_size);
	matrix theta2;
	theta2.randu(hidden_layer_size, y.n_rows);
	double eta =0.5;
	for(size_t epoch = 0; epoch < max_epochs; ++epoch)
		{
		matrix delta1,delta2;
		double cost=0.0;
		std::tie(cost,delta1,delta2) = BackProp(X,y,theta1,theta2);
		theta1 = theta1 - eta *  delta1;
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