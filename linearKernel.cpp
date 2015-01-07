#include "linearKernel.hpp"
#include "usedFun.hpp"

linearKernel::linearKernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _n = _data.cols();
  _p = _data.rows();
}
double linearKernel::squaredNorm(size_t i){
  return _data.col(i).squaredNorm();
}
void linearKernel::dot(size_t i,VectorXd & res){
  res =  _data.transpose()*_data.col(i);
}
void linearKernel::dot(vec &v,VectorXd &res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res = _data.transpose()*tmp;
}
size_t linearKernel::getN(){
  return _data.cols();
}
