#ifndef _saulZeroKernel__
#define _saulZeroKernel__
#include "Kernel.hpp"



class saulZeroKernel : public Kernel{
public:
  saulZeroKernel(matd &data,unsigned int l);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);
          
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
  virtual void dot(Ref<ArrayXd> alpha, unsigned int l);
  double calc(double angle, unsigned int l);
  virtual size_t getN();
  virtual string getName();
protected:
  size_t _p;
  size_t _n;
  unsigned int _l;
  MatrixXd _data;
  ArrayXd _dataNorm;
};
#endif
