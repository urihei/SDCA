#ifndef _learnMultiClassSVM__
#define _learnMultiClassSVM__
#include "learnSVM.hpp"



class learnMultiClassSVM: public learnSVM{
public:
  learnMultiClassSVM(ivec &y,matd &data,size_t k,
                       unsigned int iter=100,unsigned int accIter=0,
                       double lambda=1);
    virtual void learn_SDCA(mat &alpha, mat &zALPHA);
    virtual void acc_learn_SDCA(mat &alpha);
    virtual void returnModel(mat &model);
    virtual void setGamma(double gamma);
    virtual double getGamma();
protected:
    
    size_t _k;
    double _gamma;
};
#endif
