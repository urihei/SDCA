#ifndef _rbfKernelSVM__
#define _rbfKernelSVM__
#include "kernelFunctionSVM.hpp"



class rbfKernelSVM : public kernelFunctionSVM{
public:
    rbfKernelSVM(ivec &y,matd &data,size_t k,
                 double lambda=1, double gamma=1,
                 unsigned int iter=50,unsigned int accIter=0,double sigma =1);
    
    virtual double squaredNorm(size_t i);
    virtual void dot(size_t i,VectorXd &res);
    virtual void dot(vec &v,VectorXd &res);
    virtual void setSigma(double sigma);
    virtual double getSigma();
protected:
    double _sigma;
    ArrayXd _dataSquare;
};
#endif
