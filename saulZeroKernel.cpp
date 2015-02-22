#include "saulZeroKernel.hpp"
#include "usedFun.hpp"

saulZeroKernel::saulZeroKernel(matd &data, unsigned int l):_l(l){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
}
double saulZeroKernel::calc(double angle, unsigned int l){
  if(l>0){
    return dot(angle,l-1);
  }
  return 1-OneDpi*acos(angle);
}

double saulZeroKernel::dot(size_t i, size_t j){
  double angle = _data.col(j).dot(_data.col(i))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(angle,_l);
}

double saulZeroKernel::dot(vec &v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double vmNorm = vm.stableNorm();
  double angle = vm.dot(_data.col(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(angle,_l);
}
double saulZeroKernel::squaredNorm(size_t i){
  return 1;
}

void saulZeroKernel::dot(Ref<ArrayXd> alpha, unsigned int l){
  if(l>0){
    dot(alpha,l-1);
  }
  alpha = 1-OneDpi*alpha.acos();
}
void saulZeroKernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd preAn = (((_data.transpose()*_data.col(i)).array()/
                    (_dataNorm[i] * _dataNorm)).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}
void saulZeroKernel::dot(vec &v,Ref<VectorXd>res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec);
  ArrayXd preAn = (((_data.transpose()*tmp).array()/
                    (normVec * _dataNorm)).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}
void saulZeroKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
  ArrayXd preAn = (((_data.transpose()*v).array()/
                    (v.stableNorm() * _dataNorm)).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}
size_t saulZeroKernel::getN(){
  return _n;
}
string saulZeroKernel::getName(){
  char buffer[30];
  sprintf(buffer,"saulZero\t%16i",_l);
  string st(buffer);
  return st;
}
