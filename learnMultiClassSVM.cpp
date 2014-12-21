#include "learnMultiClassSVM.hpp"
#include "usedFun.hpp"

learnMultiClassSVM::learnMultiClassSVM(ivec &y,matd &data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
    :learnSVM(y,data,iter,accIter,lambda),_k(k){
    _gamma = 0.1;
}

void learnMultiClassSVM::optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a,ArrayXd &OneToK){
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
        }
    }else{
        a = (mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
    }
}
void learnMultiClassSVM::learn_SDCA(mat &alpha, mat &zALPHA){
    double lambdaN = 1/(_n*_lambda);

    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;
    unsigned int* prm = new unsigned int[_n];
    mat AlTemp(_k,_n);

    ArrayXd p(_k);
    ArrayXd mu(_k);
    ArrayXd a(_k);

    
    ArrayXd OneToK(_k);
    OneToK.setOnes();
    cumsum(OneToK.data(),_k,OneToK.data());

    VectorXd squaredNormData(_n);
    squaredNormData = _data.diagonal();
    mat zALPHAtimeK(_k,_n);
    
    zALPHAtimeK = zALPHA *_data;
    
    _data.diagonal().setZero();

    
    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        size_t curLabel = _y[i];


        p = lambdaN * (alpha * _data.col(i) + zALPHAtimeK.col(i));
        p = p - p(curLabel) +1;

        p(curLabel) = 0;        

        mu = p/(_gamma+(squaredNormData(i)*lambdaN));
        C = 1/(1+(gammaLambdan/squaredNormData(i)));

        //optimizeDual_SDCA(mu,C,a);
        optimizeDual_SDCA(mu,C,a,OneToK);
        // END optimizeDual_SDCA
        alpha.col(i) = -a;
        alpha(_y[i],i) = a.matrix().lpNorm<1>();
                
        ind++;
    }
    _data += squaredNormData.asDiagonal();
    delete prm;
}
void learnMultiClassSVM::acc_learn_SDCA(mat &alpha){
    double kappa = 100*_lambda;
    double mu = _lambda/2;
    double rho = mu+kappa;
    double eta = sqrt(mu/rho);
    double beta = (1-eta)/(1+eta);

    MatrixXd zALPHA(_k,_n);
    zALPHA.setZero();
    MatrixXd ALPHA_t(_k,_n);
    ALPHA_t.setZero();
    MatrixXd zALPHA_t(_k,_n);
    for(unsigned int t =0;t<_accIter;++t){
        learn_SDCA(alpha,zALPHA);
        zALPHA_t = zALPHA;
        zALPHA = (1+beta)*(zALPHA + alpha) - beta * ALPHA_t;
        ALPHA_t = zALPHA_t+alpha;
    }
    alpha = ALPHA_t;
}
void learnMultiClassSVM::returnModel(mat &model){

}
void learnMultiClassSVM::setGamma(double gamma){
    _gamma = gamma;
}
double learnMultiClassSVM::getGamma(){
    return _gamma;
}
