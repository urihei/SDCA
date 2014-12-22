#ifndef _SVM__
#define _SVM__

#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef vector<double> vec;
typedef vector<size_t> ivec;
typedef vector<vec> matd;
typedef MatrixXd mat;

class svm{
public:
    svm(size_t k, double lambda=1, double gamma = 1,unsigned int iter = 50, unsigned int _accIter = 0);
    virtual void learn_SDCA(mat &alpha, mat &zALPHA)=0;
    virtual void learn_acc_SDCA()=0;
    virtual void classify(matd data,ivec &res)=0;
    virtual void saveModel(string fileName)=0;

    virtual void setIter(unsigned int iter);
    virtual void setAccIter(unsigned int iter);
    virtual void setLambda(double lambda);
    virtual void setGamma(double lambda);
    
    virtual unsigned int getIter();
    virtual unsigned int getAccIter();
    virtual double getLambda();
    virtual double getGamma();
    
protected:

    //    void optimizeDual_SDCA(ArrayXd &mu,double C,mat &a,size_t i,size_t curLabel);
    void optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a);
    void fillMatrix(matd data1, mat &data2);

    unsigned int _iter; // number of iteration out loop
    unsigned int _accIter; // only when using acc_learn

    double _lambda;
    double _gamma;

    ivec _y;
    size_t _k;

    ArrayXd _OneToK;
};
#endif
