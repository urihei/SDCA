#include "polyKernelSVM.hpp"

polyKernelSVM::polyKernelSVM(ivec &y,matd &data,size_t k,
                             double lambda, double gamma,
                             unsigned int iter,unsigned int accIter,
                             double degree, double c):
    kernelFunctionSVM(y,data,k,lambda,gamma,iter,accIter),_degree(degree),_c(c){
}
double polyKernelSVM::squaredNorm(size_t i){
    return pow(_data.col(i).squaredNorm()+_c,_degree);
}
void polyKernelSVM::dot(size_t i,VectorXd & res){
    res =  ((_data.transpose()*_data.col(i)).array()+_c).pow(_degree);

}
void polyKernelSVM::dot(vec &v,VectorXd &res){
    VectorXd tmp(_p);
    for(size_t i=0; i<_p;++i){
        tmp(i) = v[i];
    }
    res =     res =  ((_data.transpose()*tmp).array()+_c).pow(_degree);
}

void polyKernelSVM::setDegree(double degree){
    _degree = degree;
}
double polyKernelSVM::getDegree(){
    return _degree;
}
void polyKernelSVM::setC(double c){
    _c = c;
}
double polyKernelSVM::getC(){
    return _c;
}

