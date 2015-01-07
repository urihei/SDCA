#ifndef _linearKernel__
#define _linearKernel__
#include "Kernel.hpp"



class linearKernel : public Kernel{
public:
  linearKernel(matd &data);
    
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,VectorXd &res);
  virtual void dot(vec &v,VectorXd &res);
  virtual size_t getN();

protected:
  size_t _p;
  size_t _n;
  MatrixXd _data;
};
#endif
