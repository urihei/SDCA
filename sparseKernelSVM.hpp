#ifndef _sparseKernelSVM__
#define _sparseKernelSVM__
#include "svm.hpp"
#include "Kernel.hpp"
#include "sparseAlpha.hpp"



class sparseKernelSVM: public svm{
public:
  sparseKernelSVM(size_t *y,Kernel* ker,size_t k,size_t n,
                  double lambda=1, double gamma=1,
                  unsigned int iter=50,unsigned int accIter=0);
  virtual double learn_SDCA();
  virtual double learn_acc_SDCA();
  
  virtual double learn_SDCA(sparseAlpha &alpha,matd &zMat,double eps);
  virtual double learn_SDCA(sparseAlpha &alpha,matd &zMat,double eps, matd& cMat);
  
  virtual void classify(matd &data,size_t* res);
  virtual void classify(double* data,size_t* res,size_t n_test,size_t p_test);
  virtual void classify(ivec_iter &itb,ivec_iter &ite,ivec &res);

  using svm::saveModel;
  virtual void saveModel(FILE* pFile);
    
  virtual void getCol(size_t i,Ref<VectorXd> kerCol);
  virtual double getGap();
  double getGap(sparseAlpha &alpha,matd &pOld,matd & res);

  virtual void init();
  virtual void setParameter(matd &par);

protected:
  sparseAlpha _alpha;
  Kernel* _ker;
  VectorXd _squaredNormData;


};
#endif
