#include <map>
#include <stdlib.h> 
#include "usedFun.hpp"

int main(int argc,char ** argv){
  std::map<size_t,double> m;

  /* initialize random seed: */
  srand (time(NULL));
  
  for(size_t i=0; i<20;++i){
    m.emplace(roll(100),rand());
  }
  for(std::map<size_t,double>::iterator it=m.begin();it!=m.end();++it){
    std::cout<<it->first<<"=>"<<it->second<<std::endl;
  }
  std::vector<std::map<size_t,double>> v(5);
  for(size_t i=0; i<20;++i){
    v[i%5].emplace(roll(100),rand());
  }
  std::map<size_t,double>::iterator it2;
  for(size_t i=0; i<5;++i){
    std::cout<<"i:"<<i;
    for(std::map<size_t,double>::iterator it=v[i].begin();it!=v[i].end();++it){
      std::cout<<it->first<<"=>"<<it->second<<" ";
      it->second = 0;
      if(i==0 && it->first > 50){
        it2 = it;
        cerr<<endl<<"II"<<it2->first<<" "<<it->first<<endl;
      }
    }
    std::cout<<std::endl;
  }
  //  cerr<<it2<<endl;
  it2->second = -3;
  for(size_t i=0; i<5;++i){
    std::cout<<"i:"<<i;
    for(std::map<size_t,double>::iterator it=v[i].begin();it!=v[i].end();++it){
      std::cout<<it->first<<"=>"<<it->second<<" ";
    }
    std::cout<<std::endl;
  }
  if(nullptr)
    std::cout<<"New"<<endl;
  return 0;
}
