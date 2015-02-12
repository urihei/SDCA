#ifndef _sparseKernelSVM__
#define _sparseKernelSVM__
#include "svm.hpp"
#include "Kernel.hpp"
#include "sparseAlpha.hpp"



class sparseKernelSVM: public svm{
public:
  sparseKernelSVM(ivec &y,Kernel* ker,size_t k,
                  double lambda=1, double gamma=1,
                  unsigned int iter=50,unsigned int accIter=0);
  virtual double learn_SDCA();
  virtual double learn_acc_SDCA();
  
  virtual double learn_SDCA(sparseAlpha &alpha,matd &zAlpha,double eps);
    
  virtual void classify(matd &data,ivec &res);
  virtual void classify(const Ref <const MatrixXd> &data,ivec &res);
  virtual void classify(ivec_iter &itb,ivec_iter &ite,ivec &res);

  using svm::saveModel;
  virtual void saveModel(FILE* pFile);
    
  virtual void getCol(size_t i,Ref<VectorXd> kerCol);
  double getGap(sparseAlpha &alpha,matd &pOld);
protected:
  sparseAlpha _alpha;
  Kernel* _ker;
  VectorXd _squaredNormData;


};
#endif
