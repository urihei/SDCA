#ifndef _learnMultiClassSVM__
#define _learnMultiClassSVM__
#include "learnSVM.hpp"



class learnMultiClassSVM: public learnSVM{
public:
    learnMultiClassSVM(int* y,vector<vector<double>> data,size_t k,
                       unsigned int iter,unsigned int accIter,
                       double lambda);
    virtual void learn_SDCA(MatrixXd &alpha, MatrixXd &zALPHA);
    virtual void acc_learn_SDCA(MatrixXd &alpha);
    virtual void returnModel(MatrixXd &model);
    virtual void setGamma(double gamma);
    virtual double getGamma();
protected:
    size_t _k;
    double _gamma;

};
#endif
