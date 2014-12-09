#include "learnMultiClassSVM.hpp"
#include "usedFun.hpp"

learnMultiClassSVM::learnMultiClassSVM(int* y,vector<vector<double>> data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
    :learnSVM(y,data,iter,accIter,lambda),_k(k){
    _gamma = 0.1;
}

void learnMultiClassSVM::learn_SDCA(MatrixXd &alpha, MatrixXd &zALPHA){
    //    double lambdaN = _n*_lambda;
    unsigned int ind = 0;
    unsigned int* prm = new unsigned int[_n];
    MatrixXd AlTemp(_n,_n);
    
    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        AlTemp = alpha;
        AlTemp.col(i).setZero();
        AlTemp = AlTemp + zALPHA;
        
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

