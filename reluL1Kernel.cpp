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
double reluL1Kernel::dot(size_t i, size_t j){
  double prod = _data.col(j).dot(_data.col(i));
  double nor = _dataNorm(i) * _dataNorm(j);
  double angle = prod/nor;
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  double delta = nor*nor - prod*prod;
  delta = (delta < 0)? 0:delta;
  return (M_PI- acos(angle))* prod + sqrt(delta) ;

}
double reluL1Kernel::dot(vec & v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double prod = vm.dot(_data.col(j));
  double nor = vm.stableNorm() * _dataNorm(j);
  double angle = prod/nor;
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  double delta = nor*nor - prod*prod;
  delta = (delta < 0)? 0:delta;
  return (M_PI- acos(angle))* prod + sqrt(delta) ;
}
  
double reluL1Kernel::squaredNorm(size_t i){
  return _dataNorm(i)*_dataNorm(i);
}
void reluL1Kernel::dot(size_t i,Ref<VectorXd>  res){
  ArrayXd prod = (_data.transpose()*_data.col(i)).array();
  ArrayXd nor = _dataNorm(i) * _dataNorm;
  res =  (M_PI-(prod/(nor)).max(-1).min(1).acos())* prod +(nor.square()-prod.square()).max(0).sqrt() ;
}
void reluL1Kernel::dot(vec &v,Ref<VectorXd> res){
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
void reluL1Kernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
    ArrayXd prod = (_data.transpose()*v).array();
    ArrayXd nor = v.stableNorm() * _dataNorm;
    res =  (M_PI-(prod/(nor)).min(1).max(-1).acos())* prod +(nor.square()-prod.square()).max(0).sqrt() ;
}
size_t reluL1Kernel::getN(){
  return _data.cols();
}

string reluL1Kernel::getName(){
    return "ReluL1";
}
