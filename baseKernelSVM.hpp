#ifndef _baseKernelSVM__
#define _baseKernelSVM__
#include "svm.hpp"




class baseKernelSVM: public svm{
public:
  baseKernelSVM(size_t k, double lambda=1, double gamma = 1,
		unsigned int iter = 50, unsigned int _accIter = 0);
  virtual double learn_SDCA();
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zAlpha);
  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zAlpha,double epsilon)=0;

  virtual void learn_acc_SDCA();
  virtual void classify(matd &data,ivec &res)=0;
  virtual void classify(ivec_iter &itb,ivec_iter &ite,ivec &res)=0;
  //    virtual void classify(mat &data,ivec &res)=0;
    
  virtual double getGap() ;
  virtual void saveModel(FILE* pFile)=0;

  virtual void setParameter(matd &par);
    
  virtual void getCol(size_t i,Ref<VectorXd> kerCol)=0;
    
protected:
  
  mat _alpha;
  double getGap(const Ref<const MatrixXd> &alpha,const Ref<const MatrixXd> &zALPHA);

};
#endif
