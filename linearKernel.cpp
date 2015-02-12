#include "linearKernel.hpp"
#include "usedFun.hpp"

linearKernel::linearKernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _n = _data.cols();
  _p = _data.rows();
}
double linearKernel::dot(size_t i, size_t j){
  return _data.col(i).dot(_data.col(j));
}
double linearKernel::dot(vec &v, size_t j){
    Map<VectorXd> vm(v.data(),_n,1);
    return vm.dot(_data.col(j));
}
double linearKernel::squaredNorm(size_t i){
  return _data.col(i).squaredNorm();
}

void linearKernel::dot(size_t i,Ref<VectorXd> res){
  res =  _data.transpose()*_data.col(i);
}
void linearKernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
  }
  res = _data.transpose()*tmp;
}
void linearKernel::dot(const Ref <const VectorXd> &v,Ref<VectorXd> res){
  res = _data.transpose()*v;
}
string linearKernel::getName(){
  return "Linear";
}

size_t linearKernel::getN(){
  return _data.cols();
}
