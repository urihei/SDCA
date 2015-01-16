#ifndef _linearSVM__
#define _linearSVM__
#include "svm.hpp"



class linearSVM: public svm{
public:
    linearSVM(ivec &y,matd &data,size_t k,
              double lambda=1, double gamma=1,
              unsigned int iter=50,unsigned int accIter=0);
    
    virtual double learn_SDCA();
    virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW);
    virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW,double eps);
    
    virtual void learn_acc_SDCA();
    virtual double getGap();
    virtual void classify(matd &data,ivec &res);
    virtual void classify(const Ref<const MatrixXd> &data,ivec &res);
  virtual void classify(ivec_iter& itb ,ivec_iter& ite ,ivec &res);    
    using svm::saveModel;
    virtual void saveModel(FILE* pFile);
    virtual void setParameter(matd &par);
protected:
    double getGap(const Ref<const MatrixXd> &alpha, const Ref<const MatrixXd>  &zW);
    size_t _p;// number of features

    mat _W;
    mat _data;
    VectorXd _squaredNormData;    
};
#endif
