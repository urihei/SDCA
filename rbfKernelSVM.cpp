#include "rbfKernelSVM.hpp"

rbfKernelSVM::rbfKernelSVM(ivec &y,matd &data,size_t k,
                           double lambda, double gamma,
                           unsigned int iter,unsigned int accIter,double sigma):
    kernelFunctionSVM(y,data,k,lambda,gamma,iter,accIter),_sigma(2*sigma){
    _dataSquare.resize(_n);
    for(size_t i=0; i<_n;++i){
        _dataSquare(i) = _data.col(i).squaredNorm();
    }

}
double rbfKernelSVM::squaredNorm(size_t i){
    return 1.0;
}
void rbfKernelSVM::dot(size_t i,VectorXd & res){
    res =  exp((2*(_data.transpose()*_data.col(i)).array() - _dataSquare[i] - _dataSquare).transpose()/_sigma);

}
void rbfKernelSVM::dot(vec &v,VectorXd &res){
    VectorXd tmp(_p);
    double squareNormData = 0.0;
    for(size_t i=0; i<_p;++i){
        tmp(i) = v[i];
        squareNormData += v[i]*v[i];
    }
    res = exp((2*(_data.transpose()*tmp).array()-squareNormData-_dataSquare)/_sigma);
}

void rbfKernelSVM::setSigma(double sigma){
    _sigma= sigma;
}
double rbfKernelSVM::getSigma(){
    return _sigma;
}
