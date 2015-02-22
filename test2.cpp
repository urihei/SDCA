#include <map>
#include <stdlib.h> 
#include "sparseAlpha.hpp"
#include "linearKernel.hpp"
#include "usedFun.hpp"

int main(int argc,char ** argv){
  sparseAlpha a1(3,10);
  sparseAlpha a2(3,10);
  
  /* initialize random seed: */
  srand (time(NULL));
  for(size_t i=0; i<3;++i){
    a1.insert(i,roll(9),roll(100));
    a1.insert(i,roll(9),roll(100));
    a2.insert(i,roll(9),roll(100));
    a2.insert(i,roll(9),roll(100));
  }
  cout<<a1.toString()<<endl;
  matd data(10);
  for(size_t i=0;i<10;++i){
    data[i].resize(5);
    for(size_t j=0;j<5;++j){
      data[i][j] = roll(100);
      fprintf(stderr,"%g ",data[i][j]);
    }
    fprintf(stderr,"\n");
  }
  linearKernel *k = new linearKernel(data); 
  vec res(3);
  vector<map<size_t,double>::iterator> indx(3);
  size_t res_val = a1.vecMul(res,1.0/8,k, 2,indx);
  for(size_t i=0;i<3;++i){
    fprintf(stderr,"%g ,",res[i]);
  }
  fprintf(stderr,"%u\n",res_val);
  vec vv{0,0,2.0,0,-1};
  size_t res_val2 = a1.vecMul(res, 1.0/8,k,vv);
  for(size_t i=0;i<3;++i){
    fprintf(stderr,"%g ,",res[i]);
  }
  fprintf(stderr,"\n%u\n",res_val2);
  cout<<endl<<a2.toString()<<endl;
  a1.add(a2);
  cout<<endl<<a1.toString()<<endl;
  exit(0);
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
