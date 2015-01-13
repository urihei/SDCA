#ifndef _zeroOneL2Kernel__
#define _zeroOneL2Kernel__
#include "Kernel.hpp"


const double OneDpi = 0.318309886183790691216444201928;
class zeroOneL2Kernel : public Kernel{
public:
    zeroOneL2Kernel(matd &data, unsigned int hidden);
    
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
  unsigned int _hidden;
  double _norm;
  vector<vector<vector<double>>> _preCalc;
};
#endif
