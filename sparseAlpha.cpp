#include <iostream>
#include "sparseAlpha.hpp"
#include <limits>

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
// return the biggest index;
size_t sparseAlpha::vecMul(vec & res, double scalar,Kernel * ker, size_t col,vector<map<size_t,double>::iterator> & indx){
  //assum v.size(0 == map.size();
  res.resize(_k);
  vec vk(_p);
  size_t big_index = _k+1;
  double val = std::numeric_limits<double>::lowest(); 
  for(size_t i=0;i<_k;++i){
    res[i] = 0;
    ker->dot(col,_alpha[i].begin(),_alpha[i].end(),vk);
    size_t ind = 0;
    for(map<size_t,double>::iterator it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      if(it->first != col || indx.empty()){
	fprintf(stderr,"%u vk %g alpha %g,\t",it->first,vk[ind], it->second);
        res[i] += vk[ind] * it->second;
      }else{
        indx[i] = it;
      }
      ind++;
    }
    fprintf(stderr,"\n");
    res[i] *= scalar;
    if(res[i] > val){
      val = res[i];
      big_index = i;
    }
  }
  return big_index;
}
size_t sparseAlpha::vecMul(vec & res, double scalar,Kernel * ker, vec & v){
  //assum v.size(0 == map.size();
  res.resize(_k);
  vec vk(_p);
  size_t big_index = _k+1;
  double val = std::numeric_limits<double>::lowest(); 
  for(size_t i=0;i<_k;++i){
    res[i] = 0;
    ker->dot(v,_alpha[i].begin(),_alpha[i].end(),vk);
    size_t ind = 0;
    for(map<size_t,double>::iterator it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      res[i] += vk[ind] * it->second;
      ind++;
    }
    res[i] *= scalar;
    if(res[i] > val){
      val = res[i];
      big_index = i;
    }
  }
  return big_index;
}

void sparseAlpha::setK(size_t k){
  _k = k;
}
void sparseAlpha::setP(size_t p){
  _p = p;
}
string sparseAlpha::toString(){
  string str;
  for(size_t i=0;i<_k;++i){
    char buff[10*_alpha[i].size()];
    int s = 0;
    for(map<size_t,double>::iterator it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      s += sprintf(buff+s,"%u %g\t",it->first,it->second);
    }
    sprintf(buff+s,"\n");
    str += buff;
  }
  return str;
}
void sparseAlpha::write(FILE* pFile){
  for(size_t i=0;i<_k;++i){
    for(map<size_t,double>::iterator it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      fprintf(pFile,"%u %g\t",it->first,it->second);
    }
    fprintf(pFile,"\n");
                
  }
}
 //add two objects;
void sparseAlpha::add(sparseAlpha a){

  for(size_t i=0;i<_k;++i){
    myMapD tmp;
    myMapD_iter tmp_it = tmp.begin();
    map<size_t,double>::iterator ita = a._alpha[i].begin();
    for(myMapD_iter it = _alpha[i].begin(); it != _alpha[i].end(); ++it){
      while(ita != a._alpha[i].end() && ita->first < it->first){
        tmp_it = tmp.emplace_hint(tmp_it,ita->first,ita->second);
        ita++;
      }
      if(ita->first == it->first){
        it->second += ita->second;
      }      
    }
    while(ita != a._alpha[i].end()){
      tmp_it = tmp.emplace_hint(tmp_it,ita->first,ita->second);
      ita++;
    }

    _alpha[i].insert(tmp.begin(),tmp.end());
    tmp.clear();
  }
}

double sparseAlpha::norm(Kernel* ker,vec & res){
  res.resize(_k);
  double val = 0;
  for(size_t k=0;k<_k;++k){
    res[k] = 0;
    for(myMapD_iter it = _alpha[k].begin(); it != _alpha[k].end(); ++it){
      // val =+ alpha(i)^2* K(i,i) - the diagonal
      res[k] += it->second * it->second * ker->squaredNorm(it->first);
      for(myMapD_iter it2 = _alpha[k].begin(); it2 != it; ++it2){
        res[k] += it->second * it2->second * ker->dot(it->first,it2->first);
      }
    }
    val += res[k];
  }
  return val;
}
//calc the norm with respect to kernel: |alpha^t K alpha |_1
double sparseAlpha::norm(Kernel* ker){
  vec tmp(_k);
  return norm(ker,tmp); 
}
