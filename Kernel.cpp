#include  "Kernel.hpp"

double Kernel::squaredNorm(size_t i){
  return dot(i,i);
}
void Kernel::dot(size_t i,Ref<VectorXd> res){
  for(size_t j=0;j<_n;++j){
    res(j) = dot(i,j);
  }
}
void Kernel::dot(vec &v,Ref<VectorXd> res){
  for(size_t j=0;j<_n;++j){
    res[j] = dot(v,j);
  }
}
virtual void dot(double* v,Ref<VectorXd> res){
 for(size_t j=0;j<_n;++j){
    res[j] = dot(v,j);
  }
}
void Kernel::dot(size_t i,map<size_t,double>::const_iterator it,
                 map<size_t,double>::const_iterator ie,vec &res){
  size_t ind =0;
  for(;it != ie; ++it){
    res[ind] = dot(i,it->first);
    ind++;
  }
}
void Kernel::dot(vec &v,map<size_t,double>::const_iterator it,
                 map<size_t,double>::const_iterator ie,vec &res){
  size_t ind =0;
  for(;it != ie; ++it){
    res[ind] = dot(v,it->first);
    ind++;
  }
}
size_t Kernel::getN(){
  return _n;
}
