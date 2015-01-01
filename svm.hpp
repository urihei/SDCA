#ifndef _SVM__
#define _SVM__

#include "def.hpp"

#include <string>
#include <iostream>

class svm{
public:
    svm(size_t k, double lambda=1, double gamma = 1,
        unsigned int iter = 50, unsigned int _accIter = 0);
    virtual double learn_SDCA(mat &alpha, mat &zALPHA)=0;
    virtual void learn_acc_SDCA()=0;
    virtual double getGap()=0;
    virtual void classify(matd &data,ivec &res)=0;
    virtual void saveModel(string fileName)=0;

    virtual void setIter(unsigned int iter);
    virtual void setAccIter(unsigned int iter);
    virtual void setCheckGap(unsigned int checkGap);
    virtual void setLambda(double lambda);
    virtual void setGamma(double lambda);
    virtual void setEpsilon(double epsilon);
    
    virtual unsigned int getIter();
    virtual unsigned int getAccIter();
    virtual unsigned int getCheckGap();
    virtual double getLambda();
    virtual double getGamma();
    virtual double getEpsilon();
    
protected:

    void optimizeDual_SDCA(ArrayXd &mu,double C,mat &a,size_t i,size_t curLabel);
    void project_SDCA(ArrayXd &mu,ArrayXd &b);
    //    void optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a);
    
    unsigned int _iter; // number of iteration out loop
    unsigned int _accIter; // only when using acc_learn

    unsigned int _chackGap; // the frequency (number of iteration) to check the gap.
    
    double _lambda;
    double _gamma;
    double _eps;
    
    ivec _y;
    size_t _k;

    ArrayXd _OneToK;
};
#endif
