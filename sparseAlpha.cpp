#include <iostream>
#include "sparseAlpha.hpp"

sparseAlpha::sparseAlpha(size_t k, size_t p):_k(k),_p(p){
  for(size_t i=0; i<_k; ++i){
    _alpha.push_back(myMapD());
  }  
}
sparseAlpha::sparseAlpha(matd m){
  _k = m.size();
  if(_k==0){
    cerr<<"The number of rows must be at least 1"<<endl;
    exit(0);
  }
  _p= m[0].size();
  for(size_t i=0; i<_k; ++i){
    _alpha.push_back(myMapD());
  }
  for(size_t i=0;i<_k;++i){
    for(size_t j=0; j< m[i].size(); ++j){
      insert(i,j,m[i][j]);
    }
  }
}
void sparseAlpha::insert(size_t row,size_t col, double val){
  if(val != 0){
    myMapD_iter alphaIter = _alpha[row].find(col);
    if(alphaIter == _alpha[row].end()){
      _alpha[row].emplace_hint(alphaIter,col,val);
    }else{
      alphaIter->second = val;
    }
  }
}
// if col(i) == 0 remove this cell.
void sparseAlpha::insert(size_t col, const vec &v){
  for(size_t i=0; i<_k;++i){
    myMapD_iter alphaIter = _alpha[i].find(col);
    if(v[i] != 0){
      if(alphaIter == _alpha[i].end()){
        _alpha[i].emplace_hint(alphaIter,col,v[i]);
      }else{
        alphaIter->second = v[i];
      }
    }else{
      if(alphaIter != _alpha[i].end()){
        _alpha[i].erase(alphaIter);
      }
    }
  }
} 

void sparseAlpha::remove(size_t row,size_t col){
  myMapD_iter alphaIter = _alpha[row].find(col);
  if(alphaIter != _alpha[row].end()){
    _alpha[row].erase(alphaIter);
  }
}
// preform res(k) = scalar * alpha(k,p) * vec(p)
void sparseAlpha::vecMul(vec & res, double scalar,Kernel * ker, size_t col,vector<map<size_t,double>::iterator> & indx){
  //assum v.size(0 == map.size();
  res.resize(_k);
  vec vk(_p);
  for(size_t i=0;i<_k;++i){
    res[i] = 0;
    ker->dot(col,_alpha[i].begin(),_alpha[i].end(),vk);
    size_t ind = 0;
    for(map<size_t,double>::iterator it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      if(it->first != col || indx.empty()){
        res[i] += vk[ind] * it->second;
      }else{
        indx[i] = it;
      }
      ind++;
    }
    res[i] *= scalar;
  }
}
void sparseAlpha::col(size_t row, vec & res) const{
}
 void sparseAlpha::setK(size_t k){
   _k = k;
 }
 void sparseAlpha::setP(size_t p){
   _p =p;
 }
