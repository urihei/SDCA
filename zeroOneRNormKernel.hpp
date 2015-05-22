#ifndef _zeroOneRNormKernel__
#define _zeroOneRNormKernel__
#include "Kernel.hpp"



class zeroOneRNormKernel : public Kernel{
public:
  zeroOneRNormKernel(matd &data, ivec &hidden);
  double calc(double alpha,unsigned int l);
  virtual double dot(size_t i, size_t j);
  virtual double dot(vec &v, size_t j);    
  virtual double squaredNorm(size_t i);
  
  void calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res,unsigned int l);
  //    void calc2(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res);
  virtual void dot(size_t i,Ref<VectorXd> res);
  virtual void dot(vec &v,Ref<VectorXd> res);
  virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
  virtual size_t getN();
  virtual string getName();
protected:
  size_t _p;
  size_t _l;
  MatrixXd _data;
  ArrayXd _dataNorm; // the norm of the data + the norm feature.
  ivec _hidden;
  vec _norm;
  double _max_norm;
  ArrayXd _normFeature;
  //vector<vector<vector<double>>> _preCalc;
  matd _preCalc;
};
#endif
