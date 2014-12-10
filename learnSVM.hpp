#ifndef _learnSVM__
#define _learnSVM__

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class learnSVM{
public:
  learnSVM(vector<int> &y,vector<vector<double>> &data,
             unsigned int iter = 100,unsigned int accIter = 0,
             double lambda = 1);
    virtual void learn_SDCA(MatrixXd &alpha, MatrixXd &zALPHA)=0;
    virtual void acc_learn_SDCA(MatrixXd &alpha)=0;
    virtual void returnModel(MatrixXd &model)=0;
    ~learnSVM();
    
protected:
    unsigned int _iter; // number of iteration out loop
    unsigned int _accIter; // only when using acc_learn
    size_t _n; // number of samples
    MatrixXd _data; // the kernek matrix _n x _n 
    double _lambda;
  vector<int> _y;

};
#endif
