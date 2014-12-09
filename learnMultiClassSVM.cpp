#include "learnMultiClassSVM.hpp"
#include "usedFun.hpp"

learnMultiClassSVM::learnMultiClassSVM(int* y,vector<vector<double>> data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
    :learnSVM(y,data,iter,accIter,lambda),_k(k){
    _gamma = 0.1;
}

void learnMultiClassSVM::learn_SDCA(MatrixXd &alpha, MatrixXd &zALPHA){
    double lambdaN = 1/(_n*_lambda);
    double gammaLambdan = _gamma*_n*_lambda;
    double C;
    unsigned int ind = 0;
    unsigned int* prm = new unsigned int[_n];
    MatrixXd AlTemp(_k,_n);

    ArrayXd p(_k);
    ArrayXd c(_k);
    ArrayXd mu(_k);
    ArrayXd a(_k);

    ArrayXd muh(_k);
    ArrayXd mub(_k);
    ArrayXd z(_k+1);
    
    ArrayXd OneToK(_k);
    OneToK.setOnes();
    cumsum(OneToK.data(),_k,OneToK.data());
    ArrayXd normOne(_k);
    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        AlTemp = alpha;
        AlTemp.col(i).setZero();
        AlTemp = AlTemp + zALPHA;

        p = lambdaN * (AlTemp * _data.col(i));
        p = p - p(_y[i]);

        c.setOnes();
        c(_y[i]) = 0;
        
        mu = (c+p)/(_gamma+_data(i,i)*lambdaN);
        C = 1/(1+gammaLambdan/_data(i,i));

        //optimizeDual_SDCA(mu,C,a);

        //creating muh
        muh = mu.max(0);
        sort(muh.data(),muh.data()+_k);
        muh.reverseInPlace();
        //creating mub
        cumsum(muh.data(),_k,mub.data());
        //creating z
        z.head(_k) = (mub - (OneToK * muh)).min(1);
        z(_k) = 1;
        //calc normOne (matlab mubDjC)
        normOne = mub/(1+(OneToK*C));
        //find indF
        size_t indF = findFirstBetween(normOne.data(),z.head(_k).data(),
                                       z.tail(_k).data(),_k);
        //
        if(indF >= _k){
            size_t indJ = findLast(z.data(),_k)
        }else{
            a = (mu+((normOne(indF)-mub(indF))/(indF+0.0))).max(0);
        }
            
        // END optimizeDual_SDCA
        alpha.col(i) = -a;
        alpha(_y[i],i) = a.matrix().lpNorm<1>();
                
        ind++;
    }

    delete prm;

}
void learnMultiClassSVM::acc_learn_SDCA(MatrixXd &alpha){

}
void learnMultiClassSVM::returnModel(MatrixXd &model){

}
void learnMultiClassSVM::setGamma(double gamma){
    _gamma = gamma;
}
double learnMultiClassSVM::getGamma(){
    return _gamma;
}

void learnMultiClassSVM::optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a){
    
}
