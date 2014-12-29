#ifndef _Kernel__
#define _Kernel__

#include <vector>
#include <Eigen/Dense>

typedef vector<double> vec;
typedef vector<size_t> ivec;
typedef vector<vec> matd;
typedef MatrixXd mat;

class Kernel{
public:
  virtual double squaredNorm(size_t i)=0;
  virtual void dot(size_t i,VectorXd &res)=0;
  virtual void dot(vecotr<double> &v,VectorXd &res)=0;
  virtual size_t getN()=0;
};
#endif
