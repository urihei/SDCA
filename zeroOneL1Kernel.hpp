#ifndef _zeroOneL1Kernel__
#define _zeroOneL1Kernel__
#include "Kernel.hpp"


const double OneDpi = 0.318309886183790691216444201928;
class zeroOneL1Kernel : public Kernel{
public:
  zeroOneL1Kernel(matd &data);
    
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,VectorXd &res);
  virtual void dot(vec &v,VectorXd &res);
  virtual size_t getN();

protected:
    size_t _p;
    size_t _n;
  MatrixXd _data;
  ArrayXd _dataNorm;
};
#endif
