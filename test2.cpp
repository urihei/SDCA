#include <map>
#include <stdlib.h> 
#include "def.hpp"
#include "usedFun.hpp"
#include "kernelSVM.hpp"
#include "linearSVM.hpp"
#include "preKernelSVM.hpp"
#include "rbfKernel.hpp"
#include "polyKernel.hpp"
#include "zeroOneL1Kernel.hpp"
#include "reluL1Kernel.hpp"
#include "zeroOneL2Kernel.hpp"
#include "zeroOneRKernel.hpp"
#include "linearKernel.hpp"
#include "saulZeroKernel.hpp"
#include "saulOneKernel.hpp"
#include "sparseAlpha.hpp"
#include "sparseKernelSVM.hpp"

int ReadData(string fileName,matd& data,ivec & label){
    string line;
    ifstream myfile;
    myfile.open(fileName.c_str(),ifstream::in);
    double tmp;
    map<int,size_t> lMap;
    unsigned int countLabel = 0;
    if (!myfile.is_open()){
        cerr << "Unable to open file" << fileName<<endl; 
        assert(false);
    }
    while ( getline (myfile,line) ){
        vec v;
        istringstream iss(line);
        while(iss){
            iss>>tmp;
            if (iss)
                v.push_back(tmp);
        }
        int cl = (int)(v.back());
        map<int,size_t>::iterator lb = lMap.lower_bound(cl);
        if(lb != lMap.end() && !(lMap.key_comp()(cl, lb->first))){
            label.push_back(lb->second);
        }else{
            label.push_back(countLabel);
            lMap.insert(lb, map<int,size_t >::value_type(cl, countLabel));
            countLabel++;
        }

        v.pop_back();

        data.push_back(v);
    }
    myfile.close();
    return countLabel;
}


int checkSparseAlpha(){
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
  fprintf(stderr,"%zu\n",res_val);
  vec vv{0,0,2.0,0,-1};
  size_t res_val2 = a1.vecMul(res, 1.0/8,k,vv);
  for(size_t i=0;i<3;++i){
    fprintf(stderr,"%g ,",res[i]);
  }
  fprintf(stderr,"\n%zu\n",res_val2);
  cout<<endl<<a2.toString()<<endl;
  a1.add(a2);
  cout<<endl<<a1.toString()<<endl;
  return 0;
}
int checkMapIterator(){
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
int main(int argc,char ** argv){
  string  fileName(argv[1]);
  matd data_t;
  ivec y_t;
  size_t k =  ReadData(fileName,data_t,y_t);
  size_t n = y_t.size();
  double lambda = 10/(n+0.0);
  Kernel* ker = new linearKernel(data_t);

  svm* svmArr[2];
  svmArr[1] = new sparseKernelSVM(y_t,ker,k,lambda,0.1,n*100,0);
  svmArr[0] = new preKernelSVM(y_t,ker,k,lambda,0.1,n*100,0);
  for(size_t i=2;i<2;++i){
    time_t start =time(NULL);
    svmArr[i]->learn_SDCA();
    cout<<i<<":\ttime :"<<time(NULL) - start<<endl;
    ivec y_res(n);
    svmArr[i]->classify(data_t,y_res);
    size_t count =0;
    for(size_t i=0;i<n;i++){
      if(y_res[i] != y_t[i])
        count++;
    }
    cout<<i<<":\tThe Number of train error "<<count<<endl;
  }
  for(size_t i=1;i<2;++i){
    svmArr[i]->setAccIter(100);
    svmArr[i]->setIter(5*n);
    time_t start =time(NULL);
    svmArr[i]->learn_acc_SDCA();
    cout<<"time :"<<time(NULL) - start<<endl;
    ivec y_res(n);
    svmArr[i]->classify(data_t,y_res);
    size_t count =0;
    for(size_t i=0;i<n;i++){
      if(y_res[i] != y_t[i])
        count++;
    }
    cout<<i<<":\tThe Number of train error "<<count<<endl;
  }
  return 0;
}
