#ifndef _rbfKernel__
#define _rbfKernel__
#include "Kernel.hpp"



class rbfKernel : public Kernel{
public:
  rbfKernel(matd &data,double sigma =1);
    
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,VectorXd &res);
  virtual void dot(vec &v,VectorXd &res);
  virtual size_t getN();

  virtual void setSigma(double sigma);
  virtual double getSigma();
protected:
    size_t _p;
    size_t _n;
  MatrixXd _data;
  double _sigma;
  ArrayXd _dataSquare;
};
#endif
