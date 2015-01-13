#ifndef _Kernel__
#define _Kernel__

#include "def.hpp"

class Kernel{
public:
    virtual double squaredNorm(size_t i)=0;
    virtual void dot(size_t i,Ref<VectorXd> res)=0;
    virtual void dot(vec &v,Ref<VectorXd> res)=0;
    virtual void dot(const Ref <const VectorXd> &v,Ref<VectorXd> res)=0;
    virtual size_t getN()=0;
    virtual string getName()=0;
    virtual ~Kernel(){}
};
#endif
