#include "kernelSVM.hpp"
#include "usedFun.hpp"

kernelSVM::kernelSVM(ivec &y, size_t k,Kernel* ker,
		     double lambda, double gamma,
		     unsigned int iter,unsigned int accIter):baseKernelSVM(k,lambda,gamma,iter,accIter){
    _y = y;
    _ker = ker;
    _n = _ker->getN();
    _alpha.resize(_k,_n);
    _squaredNormData.resize(_n);
    for(size_t i = 0;i<_n;++i){
        _squaredNormData(i) = _ker->squaredNorm(i);
    }
}
void kernelSVM::learn_SDCA(mat &alpha, mat &zALPHA){
    learn_SDCA(alpha,zALPHA,_eps);
}
void kernelSVM::learn_SDCA(mat &alpha, mat &zALPHA,double eps){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    unsigned int* prm = new unsigned int[_n];
    VectorXd kerCol(_n);
    
    
    ArrayXd p(_k);
    ArrayXd mu(_k);
    
    double gap = eps + 1;
    for(unsigned int t=1;t <= _iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        size_t curLabel = _y[i];

        _ker->dot(i,kerCol);
        alpha.col(i).setZero();

        p = lambdaN * ((alpha+zALPHA) * kerCol);

        p -= p(curLabel) - 1;

        p(curLabel) = 0;

        mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/_squaredNormData(i)));


        //note alpha is changing here
        optimizeDual_SDCA(mu,C,alpha,i,curLabel);
        
        if( t % (_n* _chackGap) == 0){
            gap = getGap(alpha,zALPHA);
            
        }
        
        if(gap < eps)
            break;
        
        ind++;
    }
    //    cerr<<alpha<<endl;
    _alpha = alpha;
    
    delete prm;
}


void kernelSVM::classify(matd &data, ivec &res){
    size_t n = data.size();
    VectorXd ya(_k);
    VectorXd kerCol(_n);
    MatrixXf::Index index;
    
    for(size_t i=0;i<n;++i){
        _ker->dot(data[i],kerCol);
        ya = _alpha * kerCol;
        ya.maxCoeff(&index);
        res[i] = (size_t) index;
    }
    
}
void kernelSVM::getCol(size_t i,VectorXd & kerCol){
    _ker->dot(i,kerCol);
}
