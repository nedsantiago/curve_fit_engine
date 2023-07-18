/* Project Title: Curve Fit Engine for Weibull Regression Modelling
 * By: Ned Santiago
 * Date Started: 2023-07-01 (Saturday, July 1, 2023)
 * Goals:
 * 1) Improve or replicate the curve fitting capabilities of the
 * closed-source program Curve Expert. Specifically, with regards to
 * the Weibull Regression Model
 * 2) Program must be able to communicate with Python Code, and be the 
 * computational engine of the curve_fit python calculator
 *
 * References:
 * Chapra, S.C., Canale, R.P. (2015). Numerical Methods for Engineers Seventh Edition.*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;

using namespace std;
using namespace Eigen;
 
double pd_a0(double x, double a0, double a1);
double pd_a1(double x, double a0, double a1);

double weibull(double x, double a, double b, double c, double d);
double pdw_a(double x, double a, double b, double c, double d);
double pdw_b(double x, double a, double b, double c, double d);
double pdw_c(double x, double a, double b, double c, double d);
double pdw_d(double x, double a, double b, double c, double d);

int main()
{
    /* Main Function*/


    // record the x-data
    double x[8] = {2, 5, 10, 15, 20, 25, 50, 100};
    const int x_len = sizeof(x) / sizeof(x[0]);
    // record the y-data
    double y[8] = {124.38122, 171.92605, 203.40486, 221.16494, 233.60009, 243.17842, 272.68472, 301.93715};

    int y_len = sizeof(y) / sizeof(y[0]);

    // Initial guesses for the values of the parameters
    double a[4] = {300, 300, 0.3, 0.4};
    int a_len = sizeof(a) / sizeof(a[0]);

    // While errors are above a certain criteria, continue algorithm
    // Errors must be below 0.01 or 1%
    double E[4] = {0,0,0,0};
    int E_len = sizeof(E) / sizeof(E[0]);
    bool is_error_acceptable = true;
    double allowable_error = 0.01;

    int iteration = 0;
    do
    {
	iteration++;
	printf("Beginning Iteration \t#%i\n",iteration);
	// algorithm goes here
	// Using a different matrix declaration
    	Matrix <double, 8, 4> Z_f;
    	// initialize to matrix of zeroes
    	Z_f.setZero();
    	printf("Initialized Z_f:\n");
    	cout << Z_f << endl;

    	// Calculate for the matrix of partial derivatives
    	for (int i = 0; i < x_len; i++)
    	{
    	    for (int j = 0; j < a_len; j++)
    	    {
		switch (j)
		{
		    case 0:
			// derive wrt a
			printf("running case 0\n");
			Z_f(i,j) = pdw_a(x[i], a[0], a[1], a[2], a[3]);
			break;
		    case 1:
			// derive wrt b
			printf("running case 1\n");
			Z_f(i,j) = pdw_b(x[i], a[0], a[1], a[2], a[3]);
			break;
		    case 2:
			// derive wrt c
			printf("running case 2\n");
			Z_f(i,j) = pdw_c(x[i], a[0], a[1], a[2], a[3]);
			break;
		    case 3:
			// derive wrt d
			printf("running case 3\n");
			Z_f(i,j) = pdw_d(x[i], a[0], a[1], a[2], a[3]);
			break;
		}
    	    }
    	}
    	printf("Calculated Z_f:\n");
    	std::cout << Z_f << std::endl;

    	// Multiply Z_f (partial derivatives) by its own transpose
    	Matrix <double, Dynamic, Dynamic> ZjT;
    	ZjT = Z_f.transpose();
    	Matrix <double, Dynamic, Dynamic> Z_inv;
    	Z_inv = ZjT * Z_f;
    	Z_inv = Z_inv.inverse();
    	// Get the result's inverse

    	printf("Calculated inverse Z_f product with transpose of Z_f\n");
    	cout << Z_inv << endl;
    	// Calculate for D (the difference between measurements and model predictions)
    	Matrix <double, x_len, 1> D;
    	D.setZero();
    	for (int i = 0; i < x_len; i++)
    	{
    	    // WHY IS IT ONLY THE FIRST COLUMN? SHOULDNT THE OTHER
    	    // PARAMETERS ALSO BE CONSIDERED??
    	    D(i,0) = y[i] - Z_f(i,0);
    	}
    	printf("D:\n");
    	cout << D << endl;

    	
    	// Multiply Z to D
    	Matrix <double, Dynamic, Dynamic> ZTD;
    	ZTD.setZero();
    	ZTD = ZjT * D;
    	printf("ZTD:\n");
    	cout << ZTD << endl;
    	
    	// Calculate for dA (delta A, change in parameters)
    	Matrix <double, Dynamic, Dynamic> dA;
    	dA.setZero();
    	dA = Z_inv * ZTD;

    	double a_temp = 0;

    	// Add the change to the previous parameters
    	for (int i = 0; i < a_len; i++)
    	{
    	    a_temp = a[i] + dA(i,0);
    	    
    	    // Record the error
    	    E[i] = (a_temp - a[i]) / a_temp;


    	    // Add the change to the previous parameters
    	    a[i] = a_temp; 
    	    
    	}
    	
    	printf("E: \n");
    	for (int i = 0; i < a_len; i++)
    	{
    	    cout << E[i] << endl;
    	}
    	printf("a: \n");
    	for (int i = 0; i < a_len; i++)
    	{
    	    cout << a[i] << endl;
    	}
    	// Redo the process until error is below a certain value
    	// Also check for the sum of squares R^2

	// check if all errors are above a certain value
	is_error_acceptable = true;
	for (int i = 0; i < E_len; i++)
	{
	    // if error is above
	    if (abs(E[i]) >= allowable_error)
	    {
		// flag as true
		printf("comparing E[%i]=%f\twith Allowable Error:%f\n", i, abs(E[i]), allowable_error);
		is_error_acceptable = false;
	    }
	    // else do nothing
	    printf("is_error_acceptable:\t%d\n", is_error_acceptable);
	}
    } while(is_error_acceptable == false);
    //while(false);
    //while(is_error_acceptable == false || iteration <= 2);

}

// original weibull formula
double weibull(double x, double a, double b, double c, double d)
{
    return a - b * exp(-c * pow(x, d));
}

// partial derivative wrt a
double pdw_a(double x, double a, double b, double c, double d)
{
    return 1;
}

// partial derivative wrt b
double pdw_b(double x, double a, double b, double c, double d)
{
    printf("running pdw_b: x=%f, a=%f, b=%f, c=%f, d=%f\n", x, a, b, c, d);
    return -exp(-c * pow(x, d));
}

// partial derivative wrt c
double pdw_c(double x, double a, double b, double c, double d)
{
    printf("running pdw_c: x=%f, a=%f, b=%f, c=%f, d=%f\n", x, a, b, c, d);
    return -b * exp(-c * pow(x, d)) * -pow(x, d);
}

// partial derivative wrt d
double pdw_d(double x, double a, double b, double c, double d)
{
    printf("running pdw_d: x=%f, a=%f, b=%f, c=%f, d=%f\n", x, a, b, c, d);
    return -b * exp(-c * pow(x, d)) * -c * d * pow(x, d - 1);
}


// partial derivative with respective to a0
double pd_a0(double x, double a0, double a1)
{
    return 1 - exp(-a1 * x);
}

// partial derivative with respective to a1
double pd_a1(double x, double a0, double a1)
{
    return a0 * x * exp(-a1 * x);
}
