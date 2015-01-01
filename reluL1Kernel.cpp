#include "reluL1Kernel.hpp"
#include "usedFun.hpp"

reluL1Kernel::reluL1Kernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
}
double reluL1Kernel::squaredNorm(size_t i){
  return _dataNorm(i)*_dataNorm(i);
}
void reluL1Kernel::dot(size_t i,VectorXd & res){
  ArrayXd prod = (_data.transpose()*_data.col(i)).array();
  ArrayXd nor = _dataNorm(i) * _dataNorm;
  res =  (M_PI-(prod/(nor)).max(-1).min(1).acos())* prod +(nor.square()-prod.square()).max(0).sqrt() ;
}
void reluL1Kernel::dot(vec &v,VectorXd &res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec);
  ArrayXd prod = (_data.transpose()*tmp).array();
  ArrayXd nor = normVec * _dataNorm;
  res =  (M_PI-(prod/(nor)).min(1).max(-1).acos())* prod +(nor.square()-prod.square()).max(0).sqrt() ;

}
size_t reluL1Kernel::getN(){
  return _data.cols();
}
