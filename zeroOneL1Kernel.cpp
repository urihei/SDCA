#include "zeroOneL1Kernel.hpp"
#include "usedFun.hpp"

zeroOneL1Kernel::zeroOneL1Kernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
}
double zeroOneL1Kernel::dot(size_t i, size_t j){
  double angle = _data.col(j).dot(_data.col(i))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return 1-OneDpi * acos(angle);
}
double zeroOneL1Kernel::dot(vec &v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double vmNorm = vm.stableNorm();
  double angle = vm.dot(_data.col(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return 1-OneDpi * acos(angle);
}
double zeroOneL1Kernel::squaredNorm(size_t i){
  return 1;
}
void zeroOneL1Kernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd preAn = (((_data.transpose()*_data.col(i)).array()/
                    (_dataNorm[i] * _dataNorm)).min(1)).max(-1);
  res =  1-OneDpi*preAn.acos();
}
void zeroOneL1Kernel::dot(vec &v,Ref<VectorXd>res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec);
  ArrayXd preAn = (((_data.transpose()*tmp).array()/
                    (normVec * _dataNorm)).min(1)).max(-1);
  res =  1-OneDpi*preAn.acos();
}
void zeroOneL1Kernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
  ArrayXd preAn = (((_data.transpose()*v).array()/
                    (v.stableNorm() * _dataNorm)).min(1)).max(-1);
  res =  1-OneDpi*preAn.acos();    
}
size_t zeroOneL1Kernel::getN(){
  return _data.cols();
}
string zeroOneL1Kernel::getName(){
  return "ZeroOneL1";
}
