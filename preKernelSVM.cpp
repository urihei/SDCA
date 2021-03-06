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
    _squaredNormData = _kernel.diagonal();
    _kernel.diagonal().setZero();
    _kerFun = NULL;
}
preKernelSVM::preKernelSVM(ivec &y,Kernel* kernel, size_t k,
          double lambda, double gamma,
          unsigned int iter,unsigned int accIter):baseKernelSVM(k,lambda,gamma,iter,accIter){
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
    //    cout<<_kernel<<endl;
    _squaredNormData = _kernel.diagonal();
    _kernel.diagonal().setZero();
    _kerFun = kernel;
}
double preKernelSVM::learn_SDCA(Ref<MatrixXd>alpha,  const Ref<const MatrixXd> &zALPHA, double eps){
    double lambdaN = 1/(_usedN*_lambda);
    
    double gammaLambdan = _gamma*_usedN*_lambda;


    double C;
    unsigned int ind = 0;

    ivec prm(_usedN);


    ArrayXd p(_k);
    ArrayXd mu(_k);
    //    ArrayXd a(_k);


    mat zALPHAtimeK(_k,_n);
    
    zALPHAtimeK = zALPHA *_kernel+zALPHA *_squaredNormData.asDiagonal();
    
    

    double gap = eps + 1;
    
    for(unsigned int t=1;t<=_iter;++t){
        if((ind % _usedN) == 0){
            randperm(_usedN,prm,_prmArray);
	    // cerr<<"_________________"<<endl;
	    // for(size_t ii=0; ii<_usedN;++ii){
	    //   cerr<<prm[ii]<<" ";
	    // }
	    
	    //cerr<<endl;
            ind = 0;
        }
        size_t i = prm[ind];
        //cerr<<i<<"@";
        size_t curLabel = _y[i];


        p = lambdaN * (alpha * _kernel.col(i) + zALPHAtimeK.col(i));
        p = p - p(curLabel) +1;

        p(curLabel) = 0;

        mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/_squaredNormData(i)));

        //optimizeDual_SDCA(mu,C,a);
        optimizeDual_SDCA(mu,C,alpha,i,curLabel);
        // END optimizeDual_SDCA
        //alpha.col(i) = -a;
        //alpha(_y[i],i) = a.matrix().lpNorm<1>();
        if( t % (_usedN* _checkGap) == 0){
            gap = getGap(alpha,zALPHA);
            
        }
        
        if(gap < eps)
            break;
        
        ind++;
    }
    //cerr<<endl;
    //    cerr<<alpha<<endl;
    
    if(gap >eps){
        gap = getGap(alpha,zALPHA);
    }
    return gap;
}


void preKernelSVM::classify(matd &data, ivec &res){
    if(_kerFun == NULL){
        MatrixXd mData;
        fillMatrix(data,mData);
        classify(mData,res);
    }else{
        size_t t_n = data.size();
        VectorXd kerCol(_n);
        VectorXd ya(_k);
        MatrixXf::Index index;
	if(_verbose)
	  cerr<<"Start kernel classify"<<endl;
        for(size_t i=0;i<t_n;++i){
            _kerFun->dot(data[i],kerCol);
            ya = _alpha * kerCol;
            ya.maxCoeff(&index);
            res[i] = (size_t) index;
        }
    }

}
void preKernelSVM::classify( const Ref<const MatrixXd> &mData, ivec &res){
    size_t n = mData.cols();
    MatrixXd ya(_k,n);
    ya = 1/(_lambda*_usedN) * (_alpha * mData);
    MatrixXf::Index index;
    for(size_t i=0;i<n;i++){
        ya.col(i).maxCoeff(&index);
        res[i] = (size_t) index;
    }
  
}
void preKernelSVM::classify(ivec_iter &itb,ivec_iter &ite,ivec &res){
  size_t n = std::distance(itb,ite);
  if(res.size() != n){
    res.resize(n);
  }
  //MatrixXd ya(_k);
  MatrixXf::Index index;
  size_t i=0;
  //  cerr<<_alpha<<endl;
  for(ivec_iter it = itb;it<ite;++it){
    VectorXd tmp = _kernel.col(*it);
    tmp(*it) = _squaredNormData(*it);
    (_alpha * tmp).array().maxCoeff(&index);
    res[i++] = (size_t) index;
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
    kerCol(i) = _squaredNormData(i);
}

