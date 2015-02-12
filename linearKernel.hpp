#ifndef _linearKernel__
#define _linearKernel__
#include "Kernel.hpp"



class linearKernel : public Kernel{
public:
  linearKernel(matd &data);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);
  
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(const Ref <const VectorXd> &v,Ref<VectorXd> res);
  virtual string getName();

  virtual size_t getN();

protected:
  size_t _p;
  size_t _n;
  MatrixXd _data;
};
#endif
