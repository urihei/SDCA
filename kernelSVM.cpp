#include "kernelSVM.hpp"
#include "usedFun.hpp"

kernelSVM::kernelSVM(ivec &y,matd &kernel, size_t k,
          double lambda, double gamma,
          unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter){
    fillMatrix(kernel,_kernel);
    _n = _kernel.cols();
    _y = y;
    _alpha.resize(_k,_n);
}
void kernelSVM::learn_SDCA(mat &alpha, mat &zALPHA){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    unsigned int* prm = new unsigned int[_n];


    ArrayXd p(_k);
    ArrayXd mu(_k);
    //    ArrayXd a(_k);

    
    VectorXd squaredNormData(_n);
    squaredNormData = _kernel.diagonal();

    mat zALPHAtimeK(_k,_n);
    
    zALPHAtimeK = zALPHA *_kernel;
    
    _kernel.diagonal().setZero();

    
    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        size_t curLabel = _y[i];


        p = lambdaN * (alpha * _kernel.col(i) + zALPHAtimeK.col(i));
        p = p - p(curLabel) +1;

        p(curLabel) = 0;

        mu = p/(_gamma+(squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/squaredNormData(i)));

        //optimizeDual_SDCA(mu,C,a);
        optimizeDual_SDCA(mu,C,alpha,i,curLabel);
        // END optimizeDual_SDCA
        //alpha.col(i) = -a;
        //alpha(_y[i],i) = a.matrix().lpNorm<1>();
                
        ind++;
    }
    _kernel += squaredNormData.asDiagonal();
    //    cerr<<alpha<<endl;
    _alpha = alpha;
    
    delete prm;
}
void kernelSVM::learn_acc_SDCA(){
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


void kernelSVM::classify(matd data, ivec &res){
    MatrixXd mData;
    fillMatrix(data,mData);
    size_t n = mData.cols();
    MatrixXd ya(_k,n);
    ya = 1/(_lambda*_n) * (_alpha * mData);
    MatrixXf::Index index;
    for(size_t i=0;i<n;i++){
        ya.col(i).maxCoeff(&index);
        res[i] = (size_t) index;
    }
  
}
void kernelSVM::saveModel(string fileName){}