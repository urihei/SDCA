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
#include "zeroOneL2Kernel.hpp"
#include "zeroOneRKernel.hpp"
#include "zeroOneRBiasKernel.hpp"
#include "saulZeroKernel.hpp"
#include "saulOneKernel.hpp"

//#include "linearKernel.hpp"






void printUsage(char* ex){
    cerr<<ex<<"\t model_file test_file"<<endl;

    exit(1);
}

void readMatrix(ifstream &myfile,matd& data){
    double tmp;
    string line;
    while ( getline (myfile,line) ){
        vec v;
        istringstream iss(line);
        while(iss){
            iss>>tmp;
            if (iss)
                v.push_back(tmp);
        }
        data.push_back(v);
    }

}
void ReadData(string fileName,matd& data){
    string line;
    ifstream myfile;
    myfile.open(fileName.c_str(),ifstream::in);
    if (!myfile.is_open()){
        cerr << "Unable to open file" << fileName<<endl; 
        assert(false);
    }
    readMatrix(myfile,data);
    myfile.close();
}

svm* ReadModel(string modelFile,Kernel* &ker,int argc ,char** argv,int* &label_map){
    svm* sv;
    ifstream myfile;
    myfile.open(modelFile.c_str(),ifstream::in);
    if (!myfile.is_open()){
        cerr << "Unable to open file" << modelFile<<endl; 
        exit(1);
    }
    string line;

    string kernel_type = "Linear";

    double lambda = 1;
    double gamma = 0.1;
    
    
    double sigma = 1; //RBF
    /*    double degree = 2;//Poly
          double c = 1; //Poly*/
    size_t* y_t;
    matd parMat;
    double* data_t;
    getline(myfile,line);
    istringstream iss_map(line);
    //cerr<<"Reading label map"<<endl;
    vector<int> tmp_map;
    int tmp;
    int indx = -1;
    while(iss_map){
        iss_map >> tmp;
        //if(tmp){
            if(indx == -1){
                indx = tmp;
            }else{
                tmp_map.push_back(tmp);
                indx = -1;
            }
            //}
    }
    label_map = new int[tmp_map.size()];
    size_t ind_map = 0;
    for(vector<int>::iterator it = tmp_map.begin(); it != tmp_map.end(); ++it){
      label_map [ind_map++] = *it;
    }
    //    cerr<<"Reading kernel parameters"<<endl;
    getline(myfile,line);
    istringstream iss(line);
    iss >> kernel_type;
    cerr<<"Kernel type: "<<kernel_type<<endl;
    if(kernel_type == "Linear"){
        iss >> lambda;
        iss >> gamma;
        readMatrix(myfile,parMat);
        sv = new linearSVM(y_t,data_t,parMat[0].size(),0,parMat.size(),lambda,gamma);
    }else{
        if( kernel_type == "preCalcKernel"){
            iss >> lambda;
            iss >> gamma;
            readMatrix(myfile,parMat);
            sv = new preKernelSVM(y_t,data_t,parMat.size(),0,lambda,gamma);
        }else{
            if(argc <4){
                cerr<<"The model is a kernel and need the train file data as the third argument"<<endl;
		exit(0);
            }
            string train_fileName(argv[3]);
            size_t n=-1;
            size_t p =-1;
	    size_t k = ReadTrainData(train_fileName,data_t,y_t,label_map,n,p);
            if(kernel_type == "RBF"){
                iss >> sigma;
                ker = new rbfKernel(data_t,n,p,sigma);
            }
            if(kernel_type == "ZeroOneRBias"){
              size_t l;
              iss >> l;
              cerr<<"L:"<<l<<"\t";
              ivec hidden_layer(l);
              for(size_t ll=0; ll<l;++ll)
                iss >> hidden_layer[ll];
              vec bias(l);
              for(size_t ll=0; ll<l;++ll)
                iss >> bias[ll];
              ker = new zeroOneRBiasKernel(data_t,n,p,hidden_layer,bias);
            }
            /*
            if(kernel_type == "Poly"){
                iss >>degree;
                iss >>c;
                ker = new polyKernel(data_t,degree,c);
            }
            if(kernel_type == "ZeroOneL1"){
                ker = new zeroOneL1Kernel(data_t);
            }
            if(kernel_type == "ReluL1"){
                ker = new reluL1Kernel(data_t);
            }
            if(kernel_type == "ZeroOneL2"){
              unsigned int hidden;
              iss >> hidden;
              ker = new zeroOneL2Kernel(data_t,hidden);
            }
            if(kernel_type == "ZeroOneR"){
              size_t l;
              iss >> l;
              ivec hidden_layer(l);
              for(size_t ll=0; ll<l;++ll)
                iss >> hidden_layer[ll];
              
              ker = new zeroOneRKernel(data_t,hidden_layer);
            }
            if(kernel_type == "saulZero"){
              size_t l;
              iss >> l;              
              ker = new saulZeroKernel(data_t,l);
            }
            if(kernel_type == "saulOne"){
              size_t l;
              iss >> l;              
              ker = new saulOneKernel(data_t,l);
              }*/
            if(ker==NULL){
                cerr<<"Unknown kernel: "<<kernel_type<<endl;
                printUsage(argv[0]);
            }
            iss >> lambda;
            iss >> gamma;
            readMatrix(myfile,parMat);

            sv = new kernelSVM(y_t,ker,k,lambda,gamma);
        }
    }
    cerr<<"Finish bulding the model"<<endl;
    sv->setParameter(parMat);
    myfile.close();
    return sv;
}
int main(int argc,char ** argv){
    if(argc < 3){
        printUsage(argv[0]);
    }
    string model_file(argv[1]);
    string test_file(argv[2]);
    
    matd testData;
    ReadData(test_file,testData);
    Kernel* ker= NULL;
    int* label_map;
    cerr<<"Reading model file"<<endl; 
    svm* sv =   ReadModel(model_file,ker,argc,argv,label_map);


    
    size_t test_size = testData.size();
    if(test_size >0 ){
      size_t y_res[testData.size()];
      cerr<<"start classify"<<endl;
        sv->classify(testData,y_res);
        cerr<<"Printing result"<<endl;
        for(size_t i=0;i<test_size;++i){
            cout<<label_map[y_res[i]]<<" ";
        }
        cout<<endl;
    }
    if(ker != NULL)
        delete ker;

    delete sv;
}
