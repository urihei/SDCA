#ifndef _preKernelSVM__
#define _preKernelSVM__
#include "baseKernelSVM.hpp"



class preKernelSVM: public baseKernelSVM{
public:
    preKernelSVM(ivec &y,matd &kernel,size_t k,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    
    virtual double learn_SDCA(mat &alpha, mat &zAlpha,double eps);
    
    virtual void classify(matd &data,ivec &res);
    using svm::saveModel;
    virtual void saveModel(string fileName);
    virtual void getCol(size_t i,VectorXd & kerCol);
    
protected:

    mat _kernel;

    
};
#endif
