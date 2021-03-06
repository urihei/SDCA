#ifndef _zeroOneL1Kernel__
#define _zeroOneL1Kernel__
#include "Kernel.hpp"



class zeroOneL1Kernel : public Kernel{
public:
  zeroOneL1Kernel(matd &data);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
  virtual size_t getN();
  virtual string getName();
protected:
  size_t _p;
  size_t _n;
  MatrixXd _data;
  ArrayXd _dataNorm;
};
#endif
