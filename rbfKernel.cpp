#include "rbfKernel.hpp"
#include "usedFun.hpp"

rbfKernel::rbfKernel(double* data,size_t n,size_t p,double sigma):
  _p(p),_sigma(2*sigma),_data(data,p,n){
  _n = n;
  _dataSquare.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataSquare(i) = _data.col(i).squaredNorm();
  }
}
double rbfKernel::dot(size_t i, size_t j){
  return exp((2*((_data.col(j)).dot(_data.col(i))) - _dataSquare[i] - _dataSquare[j])/_sigma);
}
double rbfKernel::dot(vec &v, size_t j){
  return dot(v.data(),j);
}
double rbfKernel::dot(double* v,size_t j){
  Map<VectorXd> t(v,_p,1);
  return exp((2*((_data.col(j)).dot(t)) - t.squaredNorm() - _dataSquare[j])/_sigma);
}

double rbfKernel::squaredNorm(size_t i){
  return 1.0;
}
void rbfKernel::dot(size_t i,Ref<VectorXd>  res){
  res =  exp((2*(_data.transpose()*_data.col(i)).array() - _dataSquare[i] - _dataSquare).transpose()/_sigma);

}
void rbfKernel::dot(double* v,Ref<VectorXd> res){
  Map<VectorXd> t(v,_p,1);
  res = exp((2*(_data.transpose()*t).array()-t.squaredNorm()-_dataSquare)/_sigma);
}
void rbfKernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res = exp((2*(_data.transpose()*tmp).array()-squareNormData-_dataSquare)/_sigma);
}

void rbfKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
    res = exp((2*(_data.transpose()*v).array()-v.squaredNorm()-_dataSquare)/_sigma);
}
size_t rbfKernel::getN(){
  return _data.cols();
}
void rbfKernel::setSigma(double sigma){
  _sigma= sigma;
}
double rbfKernel::getSigma(){
  return _sigma;
}
string rbfKernel::getName(){
    char buffer[30];
    sprintf(buffer,"RBF\t%16g",_sigma);
    string st(buffer);
    return st;
}
