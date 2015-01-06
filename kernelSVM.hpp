#ifndef _kernelSVM__
#define _kernelSVM__
#include "baseKernelSVM.hpp"
#include "Kernel.hpp"




class kernelSVM: public baseKernelSVM{
public:
    kernelSVM(ivec &y,size_t k,Kernel* ker,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    using baseKernelSVM::learn_SDCA;
    virtual double learn_SDCA(mat &alpha, mat &zAlpha,double eps);
    
    virtual void classify(matd &data,ivec &res);
        using svm::saveModel;
    virtual void saveModel(string fileName);
    
    virtual void getCol(size_t i,VectorXd &kerCol);
    
protected:
    Kernel* _ker;
    VectorXd _squaredNormData;


};
#endif
