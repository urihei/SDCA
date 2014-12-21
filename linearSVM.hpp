#ifndef _linearSVM__
#define _linearSVM__
#include "svm.hpp"



class linearSVM: public svm{
public:
    linearSVM(ivec &y,matd &data,size_t k,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    
    virtual void learn_SDCA(mat &alpha, mat &W);
    virtual void learn_acc_SDCA(mat &W);
    virtual ivec classify(mat data);
    virtual void saveModel(string fileName);
    
protected:

    size_t _n;
    size_t _p;// number of features


    mat _data;
    
};
#endif
