#ifndef _Kernel__
#define _Kernel__

#include "def.hpp"

class Kernel{
public:
  virtual double dot(size_t i, size_t j) =0;
  virtual double dot(vec &v, size_t j) = 0;
  virtual double dot(double* v,size_t j)=0;
  virtual double squaredNorm(size_t i);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(size_t i,map<size_t,double>::const_iterator it,
                   map<size_t,double>::const_iterator ie,vec &res);
  virtual void dot(vec &v,map<size_t,double>::const_iterator it,
                   map<size_t,double>::const_iterator ie,vec &res);
  virtual void dot(double* v,map<size_t,double>::const_iterator it,
                   map<size_t,double>::const_iterator ie,vec &res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(double* v,Ref<VectorXd> res);
  //virtual void dot(const Ref <const VectorXd> &v,Ref<VectorXd> res);
  virtual size_t getN();
  virtual string getName()=0;
  virtual ~Kernel(){}

  size_t _n;
};
#endif
