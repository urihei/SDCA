#ifndef _zeroOneRBiasKernel__
#define _zeroOneRBiasKernel__
#include "Kernel.hpp"



class zeroOneRBiasKernel : public Kernel{
public:
  zeroOneRBiasKernel(double* data, size_t n,size_t p, ivec &hidden,vec &bias);
  double calc(double alpha,unsigned int l);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);    
  virtual double dot(double* v,size_t j);
  virtual double squaredNorm(size_t i);
  
  void calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res,unsigned int l);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(double* v,Ref<VectorXd> res);
  virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
  virtual size_t getN();
  virtual string getName();
protected:
  size_t _p;
  size_t _l;
  Map<Matrix<double,Dynamic,Dynamic,ColMajor>>_data;
  ArrayXd _dataNorm;
  ivec _hidden;
  vec _norm;
  vec _bias;
  //vector<vector<vector<double>>> _preCalc;
  matd _preCalc;
};
#endif
