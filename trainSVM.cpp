#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include "usedFun.hpp"
#include <map>

#include "def.hpp"
#include "kernelSVM.hpp"
#include "linearSVM.hpp"
#include "preKernelSVM.hpp"
#include "rbfKernel.hpp"
#include "polyKernel.hpp"
#include "zeroOneL1Kernel.hpp"
#include "reluL1Kernel.hpp"
//#include "linearKernel.hpp"

enum KernelType{Linear=1,RBF=2,Poly=3,ZeroOneL1=4,ReluL1=5};


int ReadData(string fileName,matd& data,ivec & label);

void printUsage(char* ex){
    cerr<<ex<<"\tinput_file model_file KernelType[Linear|preCalcKernel|RBF|Poly|ZeroOneL1|ReluL1] lambda";
    exit(1);
}

int main(int argc,char ** argv){
    if(argc < 3){
        printUsage(argv[0]);
    }
    string data_file(argv[1]);
    string model_file(argv[2]);
    string kernel_type = "Linear";
    size_t iter = 100;
    size_t acc_iter = 0;
    double lambda = 1;
    double gamma = 0.1;
    double sigma = 1; //RBF
    
    if(argc >= 4){
        kernel_type = atoi(argv[3]);
    }
    if(argv >=5){
       lambda = atof(argv[4]);
    }

    matd data_t;
    ivec y_t;
    size_t k =  ReadData(fileName,data_t,y_t);
    size_t n = y_t.size();
    
    svm* sv;
    if(kernel_type == "Linear"){
        sv = new linearSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
    }else{
        if( kernel_type == "preCalcKernel"){
            sv = new preKernelSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
        }else{
            Kernel* ker;
            if(kernel_type == "RBF"){
                ker = new rbfKernel(data_t,sigma);
            }
            if(kernel_type == "RBF"){
                ker = new rbfKernel(data_t,sigma);
            }
        }
    }
}
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
