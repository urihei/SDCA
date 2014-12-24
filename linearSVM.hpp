#ifndef _linearSVM__
#define _linearSVM__
#include "svm.hpp"



class linearSVM: public svm{
public:
    linearSVM(ivec &y,matd &data,size_t k,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    
    virtual void learn_SDCA(mat &alpha, mat &zW);
    virtual void learn_acc_SDCA();
    virtual void classify(matd data,ivec &res);
    virtual void saveModel(string fileName);
    
protected:

    size_t _n;
    size_t _p;// number of features

    mat _W;
    mat _data;
    VectorXd _squaredNormData;    
};
#endif
