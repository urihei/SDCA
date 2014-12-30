#ifndef _kernelSVM__
#define _kernelSVM__
#include "baseKernelSVM.hpp"
#include "Kernel.hpp"




class kernelSVM: public baseKernelSVM{
public:
    kernelSVM(ivec &y,size_t k,Kernel* ker,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    virtual void learn_SDCA(mat &alpha, mat &zAlpha);
    virtual void classify(matd &data,ivec &res);

    virtual void getCol(size_t i,VectorXd &kerCol);
protected:
    Kernel* _ker;
    VectorXd _squaredNormData;

    virtual void learn_SDCA(mat &alpha, mat &zAlpha,double eps);
};
#endif
