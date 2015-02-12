#ifndef _kernelSVM__
#define _kernelSVM__
#include "baseKernelSVM.hpp"
#include "Kernel.hpp"




class kernelSVM: public baseKernelSVM{
public:
  kernelSVM(ivec &y,Kernel* ker,size_t k,
            double lambda=1, double gamma=1,
            unsigned int iter=50,unsigned int accIter=0);
  using baseKernelSVM::learn_SDCA;
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref <const MatrixXd> &zAlpha,double eps);
    
  virtual void classify(matd &data,ivec &res);
  //    virtual void classify(const Ref <const MatrixXd> &data,ivec &res);
  virtual void classify(ivec_iter &itb,ivec_iter &ite,ivec &res);
  using svm::saveModel;
  virtual void saveModel(FILE* pFile);
    
  virtual void getCol(size_t i,Ref<VectorXd> kerCol);
    
protected:
  Kernel* _ker;
  VectorXd _squaredNormData;


};
#endif
