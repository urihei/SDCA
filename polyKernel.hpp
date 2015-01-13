#ifndef _polyKernel__
#define _polyKernel__
#include "Kernel.hpp"



class polyKernel : public Kernel{
public:
    polyKernel(matd &data, double degree = 2.0,double c = 1.0 );
    
    virtual double squaredNorm(size_t i);
    virtual void dot(size_t i,Ref<VectorXd> res);
    virtual void dot(vec &v,Ref<VectorXd> res);
    virtual void dot(const Ref<const VectorXd> &v,Ref<VectorXd> res);
    
    virtual size_t getN();
    virtual string getName();
    virtual void setDegree(double sigma);
    virtual double getDegree();
    virtual void setC(double c);
    virtual double getC();
protected:
    size_t  _p;
    mat _data;
    double _degree;
    double _c;
    
};
#endif
