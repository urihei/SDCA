#include "saulOneKernel.hpp"
#include "usedFun.hpp"

saulOneKernel::saulOneKernel(matd &data, unsigned int l):_l(l){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
}
double saulOneKernel::calcAngle(double alpha, unsigned int l){
  if(l==0)
    return alpha;
  calcAngle(alpha,l-1);
  return acos((OneDpi * (sin(alpha)+(M_PI - alpha)*(cos(alpha)))));
}
double saulOneKernel::dot(size_t i, size_t j){
  double angle = _data.col(j).dot(_data.col(i))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  angle =  calcAngle(angle,_l);
  return OneDpi * _dataNorm(i)*_dataNorm(j) * ((M_PI-angle)*(cos(angle))+sin(angle));
}
double saulOneKernel::dot(vec &v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double vmNorm = vm.stableNorm();
  double angle = vm.dot(_data.col(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  angle =  calcAngle(angle,_l);
  return OneDpi * vmNorm*_dataNorm(j) * ((M_PI-angle)*(cos(angle))+sin(angle));
}
  
double saulOneKernel::squaredNorm(size_t i){
  return 1;
}

void saulOneKernel::calcAngle(Ref<ArrayXd> alpha, unsigned int l){
  if(l==0)
    return;
  calcAngle(alpha,l-1);
  alpha = (OneDpi * (alpha.sin()+(M_PI - alpha)*(alpha.cos()))).acos();
}
void saulOneKernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd preAn = ((((_data.transpose()*_data.col(i)).array()/
                    (_dataNorm[i] * _dataNorm)).min(1)).max(-1)).acos();
  calcAngle(preAn,_l);
  res =  OneDpi * _dataNorm(i)*_dataNorm * ((M_PI-preAn)*(preAn.cos())+preAn.sin());
}
void saulOneKernel::dot(vec &v,Ref<VectorXd>res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec);
  ArrayXd preAn = ((((_data.transpose()*tmp).array()/
                     (normVec * _dataNorm)).min(1)).max(-1)).acos();
  calcAngle(preAn,_l);
  res =  OneDpi * normVec*_dataNorm *((M_PI-preAn)*(preAn.cos())+preAn.sin());
}
void saulOneKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
  double vNorm = v.stableNorm();
  ArrayXd preAn = ((((_data.transpose()*v).array()/
                     ( vNorm * _dataNorm)).min(1)).max(-1)).acos();
  calcAngle(preAn,_l);
  res =  OneDpi * vNorm * _dataNorm *((M_PI-preAn)*(preAn.cos())+preAn.sin());
}
size_t saulOneKernel::getN(){
  return _n;
}
string saulOneKernel::getName(){
    char buffer[30];
    sprintf(buffer,"saulOne\t%16i",_l);
    string st(buffer);
    return st;
}
