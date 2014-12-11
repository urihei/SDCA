#ifndef _learnSVM__
#define _learnSVM__

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef vector<double> vec;
typedef vector<size_t> ivec;
typedef vector<vec> matd;
typedef MatrixXd mat;

class learnSVM{
public:
  learnSVM(ivec &y,matd &data,
             unsigned int iter = 100,unsigned int accIter = 0,
             double lambda = 1);
    virtual void learn_SDCA(mat &alpha, mat &zALPHA)=0;
    virtual void acc_learn_SDCA(mat &alpha)=0;
    virtual void returnModel(mat &model)=0;
    virtual void setIter(unsigned int iter);
    virtual void setAccIter(unsigned int iter);
    virtual void setLambda(double lambda);
    virtual unsigned int getIter();
    virtual unsigned int getAccIter();
    virtual double getLambda();
    virtual mat* getData();
    ~learnSVM();
    
protected:
    unsigned int _iter; // number of iteration out loop
    unsigned int _accIter; // only when using acc_learn
    size_t _n; // number of samples
    mat _data; // the kernek matrix _n x _n 
    double _lambda;
    ivec _y;

};
#endif
