#ifndef _kernelFunctionSVM__
#define _kernelFunctionSVM__
#include "svm.hpp"



class kernelFunctionSVM: public svm{
public:
    kernelFunctionSVM(ivec &y,matd &data,size_t k,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    
    virtual void learn_SDCA(mat &alpha, mat &zAlpha);
    virtual void learn_acc_SDCA();
    virtual void classify(matd data,ivec &res);
    virtual void saveModel(string fileName);

    virtual double squaredNorm(size_t i);
    virtual void dot(size_t i,VectorXd &res);
    virtual void dot(vec &v,VectorXd &res);
    
protected:

    size_t _n;
    size_t _p;
    
    mat _alpha;
    mat _data;
    
    VectorXd _squaredNormData;
};
#endif
