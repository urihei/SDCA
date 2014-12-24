#ifndef _polyKernelSVM__
#define _polyKernelSVM__
#include "kernelFunctionSVM.hpp"



class polyKernelSVM : public kernelFunctionSVM{
public:
    polyKernelSVM(ivec &y,matd &data,size_t k,
                 double lambda=1, double gamma=1,
                  unsigned int iter=50,unsigned int accIter=0,
                  double degree = 2.0,double c = 1.0 );
    
    virtual double squaredNorm(size_t i);
    virtual void dot(size_t i,VectorXd &res);
    virtual void dot(vec &v,VectorXd &res);
    virtual void setDegree(double sigma);
    virtual double getDegree();
    virtual void setC(double c);
    virtual double getC();
protected:
    double _degree;
    double _c;
};
#endif
