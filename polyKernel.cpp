#include "polyKernel.hpp"
#include "usedFun.hpp"

polyKernel::polyKernel(matd &data, double degree, double c):_degree(degree),_c(c){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
}

double polyKernel::squaredNorm(size_t i){
    return pow(_data.col(i).squaredNorm()+_c,_degree);
}

void polyKernel::dot(size_t i,Ref<VectorXd>  res){
    res =  ((_data.transpose()*_data.col(i)).array()+_c).pow(_degree);

}

void polyKernel::dot(vec &v,Ref<VectorXd>res){
    VectorXd tmp(_p);
    for(size_t i=0; i<_p;++i){
        tmp(i) = v[i];
    }
    res =  ((_data.transpose()*tmp).array()+_c).pow(_degree);
}
void polyKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
    res =  ((_data.transpose()*v).array()+_c).pow(_degree);
}

size_t polyKernel::getN(){
  return _data.cols();
}
void polyKernel::setDegree(double degree){
    _degree = degree;
}
double polyKernel::getDegree(){
    return _degree;
}
void polyKernel::setC(double c){
    _c = c;
}
double polyKernel::getC(){
    return _c;
}
string polyKernel::getName(){
    char buffer[50];
    sprintf(buffer,"Poly\t%20g\t%20g",_degree,_c);
    string st(buffer);
    return "Poly";
}
