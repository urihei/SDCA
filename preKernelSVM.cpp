#include "preKernelSVM.hpp"
#include "usedFun.hpp"

preKernelSVM::preKernelSVM(ivec &y,matd &kernel, size_t k,
          double lambda, double gamma,
          unsigned int iter,unsigned int accIter):baseKernelSVM(k,lambda,gamma,iter,accIter){
    fillMatrix(kernel,_kernel);
    _n = _kernel.cols();
    _usedN = _n;
    _y = y;
    _alpha.resize(_k,_n);
    _prmArray.resize(_n);
    for(size_t i=0;i<_n;++i){
        _prmArray[i] = i;
    }
    _kerFun = NULL;
}
preKernelSVM::preKernelSVM(ivec &y,Kernel* kernel, size_t k,
          double lambda, double gamma,
          unsigned int iter,unsigned int accIter):baseKernelSVM(k,lambda,gamma,iter,accIter){
    _kerFun = kernel;
    _n = kernel->getN();
    _usedN = _n;
    _y = y;
    _alpha.resize(_k,_n);
    _prmArray.resize(_n);
    cerr<<"Start create kernel"<<endl;
    
    _kernel.resize(_n,_n);
    for(size_t i=0;i<_n;++i){
        kernel->dot(i,_kernel.col(i));
        _prmArray[i] = i;
    }
}
double preKernelSVM::learn_SDCA(Ref<MatrixXd>alpha,  const Ref<const MatrixXd> &zALPHA, double eps){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    ivec prm(_usedN);


    ArrayXd p(_k);
    ArrayXd mu(_k);
    //    ArrayXd a(_k);

    
    VectorXd squaredNormData(_n);
    squaredNormData = _kernel.diagonal();

    mat zALPHAtimeK(_k,_n);
    
    zALPHAtimeK = zALPHA *_kernel;
    
    _kernel.diagonal().setZero();

    double gap = eps + 1;
    
    for(unsigned int t=1;t<=_iter;++t){
        if((ind % _usedN) == 0){
            randperm(_usedN,prm,_prmArray);
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
        if( t % (_n* _checkGap) == 0){
            gap = getGap(alpha,zALPHA);
            
        }
        
        if(gap < eps)
            break;
        
        ind++;
    }
    _kernel += squaredNormData.asDiagonal();
    //    cerr<<alpha<<endl;
    
    if(gap >eps){
        gap = getGap(alpha,zALPHA);
    }
    return gap;
}


void preKernelSVM::classify(matd &data, ivec &res){
    MatrixXd mData;
    if(_kerFun == NULL){
        fillMatrix(data,mData);
    }else{
        size_t t_n = data.size();
        mData.resize(t_n,_n);
        for(size_t i=0;i<t_n;++i){
            _kerFun->dot(data[i],mData.col(i));
        }
    }
    classify(mData,res);
}
void preKernelSVM::classify( const Ref<const MatrixXd> &mData, ivec &res){
    size_t n = mData.cols();
    MatrixXd ya(_k,n);
    ya = 1/(_lambda*_n) * (_alpha * mData);
    MatrixXf::Index index;
    for(size_t i=0;i<n;i++){
        ya.col(i).maxCoeff(&index);
        res[i] = (size_t) index;
    }
  
}
void preKernelSVM::saveModel(FILE* pFile){
    if(_kerFun == NULL){
        saveModel(pFile,"preCalcKernel",_alpha);
    }else{
        saveModel(pFile,_kerFun->getName(),_alpha);
    }
}

void preKernelSVM::getCol(size_t i,Ref<VectorXd>kerCol){
    kerCol = _kernel.col(i);
}

