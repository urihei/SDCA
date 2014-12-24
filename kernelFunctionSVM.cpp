#include "kernelFunctionSVM.hpp"
#include "usedFun.hpp"

kernelFunctionSVM::kernelFunctionSVM(ivec &y,matd &data, size_t k,
          double lambda, double gamma,
          unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter){
    fillMatrix(data,_data);
    _data.transposeInPlace();
    _n = _data.cols();
    _p = _data.rows();
    _y = y;
    _alpha.resize(_k,_n);
    _squaredNormData.resize(_n);
    for(size_t i = 0;i<_n;++i){
        _squaredNormData(i) = squaredNorm(i);
    }
}

double kernelFunctionSVM::squaredNorm(size_t i){
    return _data.col(i).squaredNorm();
}
void kernelFunctionSVM::dot(size_t i,VectorXd & res){
    res =  _data.col(i).transpose()*_data;
}
void kernelFunctionSVM::dot(vec &v,VectorXd &res){
    VectorXd tmp(_p);
    for(size_t i=0; i<_p;++i){
        tmp(i) = v[i];
    }
    res = tmp.transpose()*_data;
}

void kernelFunctionSVM::learn_SDCA(mat &alpha, mat &zALPHA){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    unsigned int* prm = new unsigned int[_n];
    VectorXd kerCol(_n);
    
    
    ArrayXd p(_k);
    ArrayXd mu(_k);


    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        size_t curLabel = _y[i];

        dot(i,kerCol);
        alpha.col(i).setZero();

        p = lambdaN * ((alpha+zALPHA) * kerCol);

        p -= p(curLabel) - 1;

        p(curLabel) = 0;

        mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/_squaredNormData(i)));

        //optimizeDual_SDCA(mu,C,a);
        //note alpha is changing here
        optimizeDual_SDCA(mu,C,alpha,i,curLabel);
        // END optimizeDual_SDCA
        //alpha.col(i) = -a;
        //alpha(_y[i],i) = a.matrix().lpNorm<1>();
                
        ind++;
    }
    //    cerr<<alpha<<endl;
    _alpha = alpha;
    
    delete prm;
}
void kernelFunctionSVM::learn_acc_SDCA(){
    double kappa = 100*_lambda;
    double mu = _lambda/2;
    double rho = mu+kappa;
    double eta = sqrt(mu/rho);
    double beta = (1-eta)/(1+eta);

    mat alpha(_k,_n);
    alpha.setZero();
    mat zALPHA(_k,_n);
    zALPHA.setZero();
    mat ALPHA_t(_k,_n);
    ALPHA_t.setZero();
    MatrixXd zALPHA_t(_k,_n);
    for(unsigned int t =0;t<_accIter;++t){
        learn_SDCA(alpha,zALPHA);
        zALPHA_t = zALPHA;
        zALPHA = (1+beta)*(zALPHA + alpha) - beta * ALPHA_t;
        ALPHA_t = zALPHA_t+alpha;
    }
    _alpha = ALPHA_t;
}


void kernelFunctionSVM::classify(matd data, ivec &res){
    size_t n = data.size();
    VectorXd ya(_k);
    VectorXd kerCol(_n);
    MatrixXf::Index index;
    
    for(size_t i=0;i<n;++i){
        dot(data[i],kerCol);
        ya = _alpha * kerCol;
        ya.maxCoeff(&index);
        res[i] = (size_t) index;
    }
    
}
void kernelFunctionSVM::saveModel(string fileName){}
