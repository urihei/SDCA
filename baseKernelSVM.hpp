#ifndef _baseKernelSVM__
#define _baseKernelSVM__
#include "svm.hpp"




class baseKernelSVM: public svm{
public:
    baseKernelSVM(size_t k, double lambda=1, double gamma = 1,
        unsigned int iter = 50, unsigned int _accIter = 0);
    virtual double learn_SDCA();
    virtual double learn_SDCA(mat &alpha, mat &zAlpha);
    virtual double learn_SDCA(mat &alpha, mat &zAlpha,double epsilon)=0;
    
    virtual void learn_acc_SDCA();
    virtual void classify(matd &data,ivec &res)=0;
    virtual double getGap() ;
    virtual void saveModel(string fileName);

    virtual void getCol(size_t i,VectorXd &kerCol)=0;
    
protected:
  
    size_t _n;
    
    mat _alpha;
    double getGap(mat &alpha,mat &zALPHA);

};
#endif
