#ifndef _learnMultiClassSVM__
#define _learnMultiClassSVM__
#include "learnSVM.hpp"



class learnMultiClassSVM: public learnSVM{
public:
  learnMultiClassSVM(ivec &y,matd &data,size_t k,
		     unsigned int iter=100,unsigned int accIter=0,
		     double lambda=1);
  virtual void learn_SDCA(mat &alpha, mat &zALPHA);
  virtual void acc_learn_SDCA(mat &alpha);
  virtual void returnModel(mat &model);
  virtual void setGamma(double gamma);
  virtual double getGamma();
protected:
  
  size_t _sRow;// number of features
  size_t _k;
  double _gamma;
  //
  ArrayXd _pp;

  ArrayXd _mu;
  ArrayXd _a;

  ArrayXd _muh;
  ArrayXd _mub;
  ArrayXd _z;

  ArrayXd _OneToK;
  
  ArrayXd _normOne;

  VectorXd _squaredNormData;
  mat _zALPHAtimeK;
  unsigned int* _prm;
  //
private:
  void buildTempVariables();
  void learn_SDCA_Iterations(mat &alpha, mat &zALPHA);
  void unBuildTempVariables();
};
#endif
