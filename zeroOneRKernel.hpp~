#ifndef _zeroOneL2Kernel__
#define _zeroOneL2Kernel__
#include "Kernel.hpp"



class zeroOneL2Kernel : public Kernel{
public:
    zeroOneL2Kernel(matd &data, unsigned int hidden);
    
    virtual double squaredNorm(size_t i);
    void calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res);
  //    void calc2(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res);
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
  unsigned int _hidden;
  double _norm;
  //vector<vector<vector<double>>> _preCalc;
  vec _preCalc;
};
#endif
