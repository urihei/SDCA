#ifndef _kernelSVM__
#define _kernelSVM__
#include "svm.hpp"



class kernelSVM: public svm{
public:
  kernelSVM(ivec &y,size_t k,Kernel &ker,
	    double lambda=1, double gamma=1,
	    unsigned int iter=50,unsigned int accIter=0);
  
  virtual void learn_SDCA(mat &alpha, mat &zAlpha);
  virtual void learn_acc_SDCA();
  virtual void classify(matd data,ivec &res);
  virtual void saveModel(string fileName);

    
protected:
  
  size_t _n;
    
  mat _alpha;
    
  VectorXd _squaredNormData;
};
#endif
