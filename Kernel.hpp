#ifndef _Kernel__
#define _Kernel__

#include "def.hpp"

class Kernel{
public:
    virtual double squaredNorm(size_t i)=0;
    virtual void dot(size_t i,VectorXd &res)=0;
    virtual void dot(vec &v,VectorXd &res)=0;
    virtual size_t getN()=0;
    virtual ~Kernel(){}
};
#endif
