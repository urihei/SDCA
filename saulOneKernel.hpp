#ifndef _saulOneKernel__
#define _saulOneKernel__
#include "Kernel.hpp"



class saulOneKernel : public Kernel{
public:
  saulOneKernel(double* data,size_t n,size_t p,unsigned int l);

  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);
  virtual double dot(double* v,size_t j);
  
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(double* v,Ref<VectorXd> res);
  virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
  virtual void calcAngle(Ref<ArrayXd> alpha, unsigned int l);
  virtual double calcAngle(double alpha, unsigned int l);
  virtual size_t getN();
  virtual string getName();
protected:
  size_t _p;
  size_t _n;
  unsigned int _l;
  Map<Matrix<double,Dynamic,Dynamic,ColMajor>>  _data;
  ArrayXd _dataNorm;
};
#endif
