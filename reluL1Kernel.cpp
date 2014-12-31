#include "reluL1Kernel.hpp"
#include "usedFun.hpp"

reluL1Kernel::reluL1Kernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataSquare.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).norm();
  }
}
double reluL1Kernel::squaredNorm(size_t i){
  return _dataNorm(i)*_dataNorm(i);
}
void reluL1Kernel::dot(size_t i,VectorXd & res){
  ArrayXd x = (_data.transpose()*_data.col(i)).array();
  res =  M_PI
}
void reluL1Kernel::dot(vec &v,VectorXd &res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res =  1-OneDpi*acos((_data.transpose()*tmp).array()/
		       (squareNormData * _dataSquare));
}
size_t reluL1Kernel::getN(){
  return _data.cols();
}
