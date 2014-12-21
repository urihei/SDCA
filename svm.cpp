#include "svm.hpp"

svm::svm(size_t k,double lambda, double gamma,unsigned int iter, unsigned int _accIter):_k(k),_lambda(lambda),_gamma(gamma),_iter(iter),_accIter(accIter){}

void svm::setIter(unsigned int iter){
    _iter = iter;
}
void svm::setAccIter(unsigned int iter){
    _accIter = iter;
}
void svm::setLambda(double lambda){
    _lambda = lambda;
}
void svm::setGamma(double gamma){
  _gamma = gamma;
}

unsigned int svm::getIter(){
    return _iter;
}
unsigned int svm::getAccIter(){
    return _accIter;
}
double svm::getLambda(){
    return _lambda;
}
double svm::getGamma(){
  return _gamma;
}

void fillMatrix(matd data1, mat &data2){
    size_t n = data1.size();
    size_t p =  data1[0].size();
    
    data2.resize(_n,p);
    
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<p;++j){
            data2(i,j) = data1[i][j];
        }
    }
}
void::optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a,size_t yi){
    ArrayXd muh(_k);
    ArrayXd mub(_k);
    ArrayXd z(_k);   
    ArrayXd normOne(_k);

    //creating muh
    muh = mu.max(0);
    sort(muh.data(),muh.data()+_k);
    muh.reverseInPlace();
    //creating mub
    cumsum(muh.data(),_k,mub.data());
    //creating z
    z  = (mub - (OneToK * muh)).min(1);
    //calc normOne (matlab mubDjC)
    normOne = mub/(1+(OneToK*C));
    //find indF
    //        size_t indF = findFirstBetween(normOne.data(),z.data(),_k);
    size_t indF = 0;
    size_t indJ = _k+1;
        
    while((indF<_k-1)&&((normOne(indF) < z(indF)) || (normOne(indF) > z(indF+1)))){
        if(indJ == _k+1 && z(indF) ==1){
            indJ = indF -1;
        }
        indF++;
    }
    if((indF==_k-1)&& !((normOne(indF)>=z(indF))&& (normOne(indF) <= 1)) ){
        if(indJ == _k+1 && z(indF) ==1){
            indJ = indF -1;
        }
        indF++;
    }

    if(indF >= _k){
        // size_t indJ = findFirst(z.data(),_k)-1;
        a = (mu+((1-mub(indJ))/(indJ+1))).max(0);

        if(((a-mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
            a.setZero();
        }else{
            a = -a;
            a(yi) = 1;
        }
    }else{
        a = -(mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
        a(yi) = normOne(indF);
    }
}
