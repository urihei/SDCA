#include "linearSVM.hpp"
#include "usedFun.hpp"

linearSVM::linearSVM(ivec &y,matd &data,size_t k,
                     double lambda, double gamma,
                     unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter){
    fillMatrix(data,_data);
    _data.transposeInPlace();
    _n = _data.cols();
    _p = _data.rows();
    _y = y;
    _W.resize(_p,_k);
    _squaredNormData.resize(_n);
    for(size_t i = 0;i<_n;++i){
        _squaredNormData(i) = _data.col(i).squaredNorm();
    }
}
void linearSVM::learn_SDCA(mat &alpha, mat &zW){


    double lambdaN = 1/(_n*_lambda);
    double gammaLambdan = _gamma*_n*_lambda;


    unsigned int* prm = new unsigned int[_n];
  

    unsigned int ind = 0;

    ArrayXd p(_k);
    ArrayXd mu(_k);
    ArrayXd a(_k);
    double C;
  
  
    _W = zW + lambdaN * _data *alpha.transpose(); 

    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        size_t curLabel = _y[i];

        _W -= lambdaN * _data.col(i)*alpha.col(i).transpose();
                
        p = _W.transpose() * _data.col(i);
        p -= p(curLabel) -1;
        p(curLabel) = 0;



        mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/_squaredNormData(i)));



        //    optimizeDual_SDCA(mu,C,a);
        optimizeDual_SDCA(mu,C,alpha,i,curLabel);
        // END optimizeDual_SDCA

        //alpha.col(i)     = - a;
        //alpha(curLabel,i) = a.matrix().lpNorm<1>();

        
        _W += lambdaN * _data.col(i)*alpha.col(i).transpose(); 

        ind++;
    }
}

void linearSVM::learn_acc_SDCA(){
    double kappa = 100*_lambda;
    double mu = _lambda/2;
    double rho = mu+kappa;
    double eta = sqrt(mu/rho);
    double beta = (1.0-eta)/(1.0+eta);

    mat alpha(_k,_n);
    alpha.setZero();

    mat zW(_p,_k);
    zW.setZero();
    
    mat W_t(_p,_k);
    W_t.setZero();

    for(unsigned int t =0; t<_accIter; ++t){
        learn_SDCA(alpha, zW);
        zW = (1+beta)*_W - beta * W_t;
        W_t = _W;
    }
}
void linearSVM::classify(matd data,ivec &res){
    MatrixXd mData;
    fillMatrix(data,mData);
    mData.transposeInPlace();
    size_t n = mData.cols();
    MatrixXd ya(_k,n);
    ya = _W.transpose()*_data;
    MatrixXf::Index index;
    for(size_t i=0;i<n;i++){
        ya.col(i).maxCoeff(&index);
        res[i] = (size_t) index;
    }
}
void linearSVM::saveModel(string fileName){
}

