#include "kernelSVM.hpp"
#include "usedFun.hpp"

kernelSVM::kernelSVM(ivec &y,Kernel* ker, size_t k,
		     double lambda, double gamma,
		     unsigned int iter,unsigned int accIter):baseKernelSVM(k,lambda,gamma,iter,accIter){
    _y = y;//reassign;
    _ker = ker;
    _n = _ker->getN();
    _usedN = _n;
    _alpha.resize(_k,_n);
    _squaredNormData.resize(_n);
    _prmArray.resize(_n);
    for(size_t i = 0;i<_n;++i){
        _squaredNormData(i) = _ker->squaredNorm(i);
        _prmArray[i] = i;
    }
}
double kernelSVM::learn_SDCA(Ref <MatrixXd> alpha, const Ref <const MatrixXd> &zALPHA,double eps){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    ivec prm(_usedN);
    VectorXd kerCol(_n);
    
    
    ArrayXd p(_k);
    ArrayXd mu(_k);
    
    double gap = eps + 1;
    for(unsigned int t=1;t <= _iter;++t){
        if((ind % _usedN) == 0){
            randperm(_usedN,prm,_prmArray);
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
        
        if( t % (_n* _checkGap) == 0){
            gap = getGap(alpha,zALPHA);
            
        }
        
        if(gap < eps)
            break;
        
        ind++;
    }

    if(gap >eps){
        gap = getGap(alpha,zALPHA);
    }
    return gap;
}


void kernelSVM::classify(matd &data, ivec &res){
    size_t n = data.size();
    VectorXd ya(_k);
    VectorXd kerCol(_n);
    MatrixXf::Index index;
    cerr<<"Start kernel classify"<<endl;
    for(size_t i=0;i<n;++i){
        _ker->dot(data[i],kerCol);
        ya = _alpha * kerCol;
        ya.maxCoeff(&index);
        res[i] = (size_t) index;
    }
    
}
void kernelSVM::classify(ivec_iter &itb,ivec_iter &ite,ivec &res){
  size_t n = std::distance(itb,ite);
  if(res.size() != n){
    res.resize(n);
  }
  VectorXd kerCol(_n);
  MatrixXf::Index index;
  size_t i =0;
  for(ivec_iter it =itb; it<ite;++it){
    _ker->dot(*it,kerCol);
    (_alpha * kerCol).maxCoeff(&index);
    res[i++] = (size_t) index;
  }
}
void kernelSVM::classify(const Ref <const MatrixXd> &data,ivec &res){
    size_t n = data.cols();
    VectorXd ya(_k);
    VectorXd kerCol(_n);
    MatrixXf::Index index;
    
    for(size_t i=0;i<n;++i){
        _ker->dot(data.col(i),kerCol);
        ya = _alpha * kerCol;
        ya.maxCoeff(&index);
        res[i] = (size_t) index;
    }
}
void kernelSVM::saveModel(FILE* pFile){
    saveModel(pFile,_ker->getName(),_alpha);
}
void kernelSVM::getCol(size_t i,Ref <VectorXd>  kerCol){
    _ker->dot(i,kerCol);
}
