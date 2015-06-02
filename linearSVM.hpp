#ifndef _linearSVM__
#define _linearSVM__
#include "svm.hpp"



class linearSVM: public svm{
public:
  linearSVM(size_t* y,double* data,size_t k,size_t n,size_t p,
            double lambda=1, double gamma=1,
            unsigned int iter=50,unsigned int accIter=0);
    
  virtual double learn_SDCA();
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW);
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW,double eps);
    
  virtual double learn_acc_SDCA();
  virtual double getGap();
  virtual void classify(double* data,size_t* res,size_t n_test,size_t p_test);
  virtual void classify(matd &data,size_t* res);
  virtual void classify(const Ref<const MatrixXd> &data,size_t* res);
  virtual void classify(ivec_iter& itb ,ivec_iter& ite ,ivec &res);    

  using svm::saveModel;
  virtual void saveModel(FILE* pFile);
  virtual void setParameter(matd &par);
  virtual void init();
protected:
  double getGap(const Ref<const MatrixXd> &alpha, const Ref<const MatrixXd>  &zW);
  size_t _p;// number of features

  mat _W;
  Map<Matrix<double,Dynamic,Dynamic,ColMajor>> _data;
  VectorXd _squaredNormData;    
};
#endif
