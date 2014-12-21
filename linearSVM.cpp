#include "linearSVM.hpp"
#include "usedFun.hpp"
linearSVM::linearSVM(ivec &y,matd &data,size_t k,
                     double lambda, double gamma,
                     unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter),_y(y){
    fillMatrix(data,_data);
    _n = data.cols();
    _p = data.rows();
}
void linearSVM::learn_SDCA(mat &alpha, mat &W){


    double lambdaN = 1/(_n*_lambda);
    double gammaLambdan = _gamma*_n*_lambda;


    unsigned int* prm = new unsigned int[_n];
  

    unsigned int ind = 0;

    ArrayXd p(_k);
    ArrayXd mu(_k);
    //  ArrayXd a(_k);
    double C;
  
    VectorXd squaredNormData(_n);
    squaredNormData = _data.diagonal();
  
    W += lambdaN * _data *alpha.transpose(); 

    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,_prm);
            ind = 0;
        }
        size_t i = _prm[ind];
        size_t curLabel = _y[i];
        W -= lambdaN * _data.col(i)*alpha.col(i).transpose();
        p = W.transpose() * _data.col(i);
        p -=  p(curLabel) +1;
        p(curLabel) = 0;

        //
        mu = p/(_gamma+(squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/quaredNormData(i)));

        //optimizeDual_SDCA(mu,C,a);
        optimizeDual_SDCA(mu,C,alpha.col(i),curLabel);
        // END optimizeDual_SDCA

        //alpha.col(i)     = - a;
        //alpha(curLabel,i) = a.matrix().lpNorm<1>();
    
        W += lambdaN * _data.col(i)*alpha.col(i).transpose(); 
                
        ind++;
    }
}

void linearSVM::learn_acc_SDCA(mat &zW){
    double kappa = 100*_lambda;
    double mu = _lambda/2;
    double rho = mu+kappa;
    double eta = sqrt(mu/rho);
    double beta = (1-eta)/(1+eta);

    MatrixXd alpha(_k,_n);
    alpha.setZero();
    MatrixXd W_t(_p,_k);
    MatrixXd W_tt(_p,_k);
    W_t.setZero();

    for(unsigned int t =0; t<_accIter; ++t){
        learn_SDCA_Iterations(alpha, zW);
        W_tt = zW;
        zW = (1+beta)*zW - beta * W_t;
        W_t = W_tt;
    }
    zW = W_t;

}
ivec linearSVM::classify(mat data){}
void linearSVM::saveModel(string fileName){}

