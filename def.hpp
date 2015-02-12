#ifndef _def__
#define _def__
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef vector<double> vec;
typedef vector<size_t> ivec;
typedef vector<size_t>::iterator ivec_iter;
typedef vector<vec> matd;
typedef MatrixXd mat;

typedef map<size_t,double> myMapD;
typedef map<size_t,double>::iterator myMapD_iter;
typedef map<size_t,unsigned int> myMapUI;
typedef map<size_t,unsigned int>::iterator myMapUI_iter;


const double OneDpi = 0.318309886183790691216444201928;
const double eps = 1e-16;
#endif

