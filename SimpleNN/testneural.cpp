#include <iostream>
#include <mlpack/core.hpp>
#include <tuple>
#include <cstdlib>     /* srand, rand */
#include <ctime>       /* time */
#include "neural.hpp"
	

void main()
	{
	// Load the dataset.
	using namespace mlpack;
	arma::mat dataset;
	data::Load("thyroid_train.csv", dataset, true);
	srand(2153);
	arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
		dataset.n_cols - 1);
	arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0,
		dataset.n_rows - 1, dataset.n_cols - 1);
	matrix theta1, theta2;
	std::tie(theta1, theta2) = TrainNetwork(trainData, trainLabels, 4);
	size_t m = trainData.n_cols;
	matrix a1 = arma::join_cols(arma::ones(1, m), trainData);
	auto trainResult = Predict(a1, theta1, theta2);
	double error = ComputeError(trainResult, trainLabels);
	std::cout << "Error: " << error << std::endl;
	data::Load("thyroid_test.csv", dataset, true);

	arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
		dataset.n_cols - 1);
	arma::mat testLabels = dataset.submat(dataset.n_rows - 3, 0,
		dataset.n_rows - 1, dataset.n_cols - 1);
	matrix a2 = arma::join_cols(arma::ones(1, testData.n_cols), testData);
	auto testResult = Predict(a2, theta1, theta2);
	error = ComputeError(testResult, testLabels);
	std::cout << "Error: " << error << "/" << testLabels.n_cols << std::endl;
	}