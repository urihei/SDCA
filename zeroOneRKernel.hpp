#ifndef _zeroOneRKernel__
#define _zeroOneRKernel__
#include "Kernel.hpp"



class zeroOneRKernel : public Kernel{
public:
  zeroOneRKernel(matd &data, ivec &hidden);
    
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
  size_t _n;
  size_t _l;
  MatrixXd _data;
  ArrayXd _dataNorm;
  ivec _hidden;
  vec _norm;
  //vector<vector<vector<double>>> _preCalc;
  matd _preCalc;
};
#endif
