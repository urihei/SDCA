#ifndef _preKernelSVM__
#define _preKernelSVM__
#include "baseKernelSVM.hpp"
#include "Kernel.hpp"


class preKernelSVM: public baseKernelSVM{
public:
  preKernelSVM(size_t* y,double* kernel,size_t k,size_t n,
               double lambda=1, double gamma=1,
               unsigned int iter=50,unsigned int accIter=0);
  preKernelSVM(size_t* y,Kernel* kernel,size_t k,size_t n,
               double lambda=1, double gamma=1,
               unsigned int iter=50,unsigned int accIter=0);
  using baseKernelSVM::learn_SDCA;
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zAlpha,double eps);
    
  virtual void classify(double* data,size_t* res,size_t n_test,size_t p_test);
  virtual void classify(matd &data,size_t* res);
  virtual void classify(const Ref<const MatrixXd> &data,size_t* res);
  virtual void classify(ivec_iter& itb ,ivec_iter& ite ,ivec &res);    

  using svm::saveModel;
  virtual void saveModel(FILE* pFile);
  virtual void getCol(size_t i,Ref<VectorXd> kerCol);
    
protected:

  //  Map<Matrix<double,Dynamic,Dynamic,ColMajor>> _kernel;
  MatrixXd _kernel;
  Kernel* _kerFun;
  VectorXd _squaredNormData;//need to fix
    
};
#endif
