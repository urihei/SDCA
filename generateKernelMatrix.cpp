#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <map>

#include "def.hpp"
#include "usedFun.hpp"
#include "rbfKernel.hpp"
#include "polyKernel.hpp"
#include "zeroOneL1Kernel.hpp"
#include "reluL1Kernel.hpp"
#include "zeroOneL2Kernel.hpp"
#include "zeroOneRKernel.hpp"
#include "zeroOneRNormKernel.hpp"
#include "zeroOneRBiasKernel.hpp"
#include "saulZeroKernel.hpp"
#include "saulOneKernel.hpp"
#include "linearKernel.hpp"

void printUsage(char* ex){
  cerr<<ex<<"\t input_file out_file"<<endl;
  cerr<<"\t Kernel Option:"<<endl;
  cerr<<"\t\t"<<"-sigma value[double]"<<"\t default 1"<<" Sigam for the RBF kernel"<<endl;
  cerr<<"\t\t"<<"-degree value[double]"<<"\t default 2"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
  cerr<<"\t\t"<<"-c value[double]"<<"\t default 1"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
  cerr<<"\t\t"<<"-hidden value[unsigned int]"<<"\t default 20"<<" Number of hidden units in the kernel"<<endl; 
}

int main(int argc,char ** argv){
  if(argc < 2 || argc % 2 != 1){
    printUsage(argv[0]);
  }
  string data_file(argv[1]);
  string out_file(argv[2]);
  
  string kernel_type = "RBF";
  double sigma = 1; //RBF
  /*  double degree = 2;//Poly
  double c = 1; //Poly
  unsigned int hidden = 5;
  unsigned int l = 0;*/
  ivec hidden_layer {1,20,10};
  vec bias {1.0,1.0,1.0};
  
  for(int i=3;i<argc; i+= 2){
    bool rec = false;
    if(strcmp("-k",argv[i])==0){
      kernel_type  = string(argv[i+1]);
      rec = true;
    }
    if(strcmp("-sigma",argv[i])==0){
      sigma  = atof(argv[i+1]);
      rec = true;
    }
    if(strcmp("-hidden_layer",argv[i])==0){
      stringstream ss(argv[i+1]);
      string item;
      hidden_layer.clear();
      hidden_layer.push_back(1);
      getline(ss,item,'-');
      while(getline(ss,item,'-')){
        hidden_layer.push_back(atoi(item.c_str()));
      }
      rec = true;
    }
    if(strcmp("-bias",argv[i])==0){
      stringstream ss(argv[i+1]);
      bias.clear();
      string item;
      while(getline(ss,item,'-')){
        bias.push_back(stod(item));
      }
      rec = true;
    }      
    
    /*   if(strcmp("-degree",argv[i])==0){
      degree  = atof(argv[i+1]);
      rec = true;
    }
    if(strcmp("-c",argv[i])==0){
      c  = atof(argv[i+1]);
      rec = true;
    }
    if(strcmp("-hidden",argv[i])==0){
      hidden  = atoi(argv[i+1]);
      rec = true;
      }
    if(strcmp("-l",argv[i])==0){
      l  = atoi(argv[i+1]);
      rec = true;
    }*/
    if(! rec){
      cerr<<"option "<<argv[i]<<" is not recognized"<<endl;
      printUsage(argv[0]);
    }
  }
  
  double* data_t;
  size_t* y_t;
  int* label_map;
  size_t n = -1;
  size_t p = -1;
  ReadTrainData(data_file,data_t,y_t,label_map,n,p);
  Kernel* ker= NULL;
  if(kernel_type == "RBF"){
    ker = new rbfKernel(data_t,n,p,sigma);
  }
  if(kernel_type == "ZeroOneRBias"){
    ker = new zeroOneRBiasKernel(data_t,n,p,hidden_layer,bias);
  }
  /*
  if(kernel_type == "Poly"){
    ker = new polyKernel(data_t,degree,c);
            }
  if(kernel_type == "ZeroOneL1"){
    ker = new zeroOneL1Kernel(data_t);
  }
  if(kernel_type == "ReluL1"){
    ker = new reluL1Kernel(data_t);
  }
  if(kernel_type == "ZeroOneL2"){
                ker = new zeroOneL2Kernel(data_t,hidden);
  }
  if(kernel_type == "ZeroOneR"){
    ker = new zeroOneRKernel(data_t,hidden_layer);
  }
  if(kernel_type == "ZeroOneRNorm"){
    ker = new zeroOneRNormKernel(data_t,hidden_layer);
  }
  if(kernel_type == "saulZero"){
        ker = new saulZeroKernel(data_t,l);
  }
  if(kernel_type == "saulOne"){
    ker = new saulOneKernel(data_t,l);
    }*/
  if(ker==NULL){
    cerr<<"Unknown kernel: "<<kernel_type<<endl;
    printUsage(argv[0]);
  }
  MatrixXd _kernel(n,n);
  for(size_t i=0;i<n;++i){
    ker->dot(i,_kernel.col(i));
  }
  FILE* pFile = fopen(out_file.c_str(),"w");
  for(size_t i=0;i<n;++i){
    for(size_t j=0;j<n;++j){
      fprintf(pFile,"%g ",_kernel(i,j));
    }
    fprintf(pFile,"%i\n",label_map[y_t[i]]);
  }
  fclose(pFile);
  return 0;
}
