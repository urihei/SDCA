#ifndef _rbfKernel__
#define _rbfKernel__
#include "Kernel.hpp"



class rbfKernel : public Kernel{
public:
  rbfKernel(double* data,size_t n,size_t p,double sigma =1);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);
  virtual double dot(double* v,size_t j);

  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(double* v,Ref<VectorXd> res);    

  virtual void dot(const Ref<const VectorXd> &v,Ref<VectorXd> res);
  
  virtual size_t getN();
  virtual string getName();
  virtual void setSigma(double sigma);
  virtual double getSigma();
protected:
  size_t _p;
  double _sigma;
  Map<Matrix<double,Dynamic,Dynamic,ColMajor>>_data;
  ArrayXd _dataSquare;
};
#endif
