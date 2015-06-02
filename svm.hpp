#ifndef _SVM__
#define _SVM__

#include "def.hpp"

#include <string>
#include <iostream>
#include <stdio.h>

class svm{
public:
  svm(size_t k, double lambda=1, double gamma = 1,
      unsigned int iter = 50, unsigned int _accIter = 0);
  virtual ~svm();
  //  virtual double learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zALPHA)=0;
  virtual double learn_SDCA()=0;
  virtual double learn_acc_SDCA()=0;

  virtual double getGap()=0;
  
  virtual void classify(matd &data,size_t* res)=0;
  virtual void classify(double* data,size_t* res,size_t n_test,size_t p_test)=0;
  virtual void classify(ivec_iter &itb,ivec_iter &ite,ivec &res)=0; // classify part of train data.
    
  virtual void saveModel(FILE* pFile)=0;
  virtual void saveModel(FILE* pFile, string kernel,const Ref<const MatrixXd> & model);
  virtual void saveModel(FILE* pFile, string kernel,const matd & model);

  virtual void setParameter(matd &par) = 0;

  virtual void setIter(unsigned int iter);
  virtual void setAccIter(unsigned int iter);

  virtual void init()=0;

  virtual void setCheckGap(unsigned int checkGap);
  virtual void setCheckGapAcc(unsigned int checkGap);
  virtual void setLambda(double lambda);
  virtual void setGamma(double lambda);
  virtual void setEpsilon(double epsilon);
  virtual void setVerbose(bool ver);
  virtual void setUsedN(size_t n);

  virtual void samplePrm();
  virtual void shiftPrm(size_t n);
  virtual ivec_iter getPrmArrayBegin();
  virtual ivec_iter getPrmArrayEnd();
  
  virtual size_t getN();
  virtual unsigned int getIter();
  virtual unsigned int getAccIter();
  virtual unsigned int getCheckGap();
  virtual unsigned int getCheckGapAcc();
  virtual double getLambda();
  virtual double getGamma();
  virtual double getEpsilon();
  virtual bool getVerbose();
    
protected:

  void optimizeDual_SDCA(const Ref<const ArrayXd> &mu,double C,Ref<MatrixXd> a,size_t i,size_t curLabel);
  double optimizeDual_SDCA(vec &mu,double C,vec &a);
  void project_SDCA(const Ref<const ArrayXd> &mu,Ref<ArrayXd> b);
  double project_SDCA(vec &mu,vec &b);
  //    void optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a);
    
  unsigned int _iter; // number of iteration out loop
  unsigned int _accIter; // only when using acc_learn

  unsigned int _checkGap; // the frequency (number of iteration) to check the gap.
  unsigned int _checkGapAcc; // the frequency (number of iteration) to check the gap in accelerated SDCA.
    
  double _lambda;
  double _gamma;
  double _eps;

  bool _verbose;
    
  size_t* _y;
  size_t _k;
  size_t _n;

  size_t _usedN;
  ivec _prmArray;
    
  ArrayXd _OneToK;
};
#endif
