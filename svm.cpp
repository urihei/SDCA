#include "svm.hpp"
#include "usedFun.hpp"

svm::svm(size_t k,double lambda, double gamma,unsigned int iter, unsigned int accIter):_iter(iter),_accIter(accIter),_lambda(lambda),_gamma(gamma),_k(k){
    _OneToK.resize(_k);
    _OneToK.setOnes();
    cumsum(_OneToK.data(),_k,_OneToK.data());
    _eps = 1e-3;
    _checkGap = 5;
    _checkGapAcc = 5;
    _verbose = false;
}

void svm::setIter(unsigned int iter){
    _iter = iter;
}
void svm::setAccIter(unsigned int iter){
    _accIter = iter;
}
void svm::setCheckGap(unsigned int checkGap){
    _checkGap = checkGap;
}
void svm::setCheckGapAcc(unsigned int check){
    _checkGapAcc = check;
}
void svm::setLambda(double lambda){
    _lambda = lambda;
}
void svm::setGamma(double gamma){
    _gamma = gamma;
}
void svm::setEpsilon(double epsilon){
    _eps = epsilon;
}
void svm::setVerbose(bool ver){
    _verbose = ver;
}

unsigned int svm::getIter(){
    return _iter;
}
unsigned int svm::getAccIter(){
    return _accIter;
}
unsigned int svm::getCheckGap(){
    return _checkGap;
}
unsigned int svm::getCheckGapAcc(){
    return _checkGapAcc;
}
double svm::getLambda(){
    return _lambda;
}
double svm::getGamma(){
    return _gamma;
}
double svm::getEpsilon(){
    return _eps;
}
bool svm::getVerbose(){
    return _verbose;
}

void svm::optimizeDual_SDCA(ArrayXd &mu,double C,mat &a,size_t i,size_t yi){
    //void svm::optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a){
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
    z  = (mub - (_OneToK * muh)).min(1);
    //calc normOne (matlab mubDjC)
    normOne = mub/(1+(_OneToK*C));
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
        a.col(i) = (mu+((1-mub(indJ))/(indJ+1))).max(0);
        //a  = (mu+((1-mub(indJ))/(indJ+1))).max(0);
        if(((a.col(i).array() - mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
            //if(((a - mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
            a.col(i).setZero();
            //a.setZero();
        }else{
            a.col(i) = -1* a.col(i);
            a(yi,i) = 1;
        }
    }else{
        a.col(i) = -1*(mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
        //a = (mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
        a(yi,i) = normOne(indF);
    }
}
void svm::project_SDCA(ArrayXd &mu,ArrayXd &b){
    ArrayXd muh(_k);
    ArrayXd mub(_k);
    ArrayXd z(_k);   
    //
    muh = mu.max(0);
    if(muh.sum() <= 1){
        b = muh;
        return;
    }
    sort(muh.data(),muh.data()+_k);
    muh.reverseInPlace();
    cumsum(muh.data(),_k,mub.data());

    z = 1 + _OneToK * muh - mub;

    size_t ind = _k-1;
    while( ind > 0 && z(ind)<= 0){
        ind --;
    }
    b = mu+ (1-mub(ind))/(ind+1);
    b = b.max(0);
}
