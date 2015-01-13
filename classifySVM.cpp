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
    map<int,size_t> lMap;
    if (!myfile.is_open()){
        cerr << "Unable to open file" << fileName<<endl; 
        assert(false);
    }
    readMatrix(myfile,data);
    myfile.close();
}
int ReadTrainData(string fileName,matd& data,ivec & label){
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
svm* ReadModel(string modelFile,Kernel* &ker,int argc ,char** argv,vector<int> &label_map){
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
    double degree = 2;//Poly
    double c = 1; //Poly
    ivec y_t;
    matd parMat;
    matd data_t;
    getline(myfile,line);
    istringstream iss_map(line);
    //cerr<<"Reading label map"<<endl;
    int tmp;
    int indx = -1;
    while(iss_map){
        iss_map >> tmp;
        //if(tmp){
            if(indx == -1){
                indx = tmp;
            }else{
                label_map.push_back(tmp);
                indx = -1;
            }
            //}
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
        sv = new linearSVM(y_t,data_t,parMat[0].size(),lambda,gamma);
    }else{
        if( kernel_type == "preCalcKernel"){
            iss >> lambda;
            iss >> gamma;
            readMatrix(myfile,parMat);
            sv = new preKernelSVM(y_t,data_t,parMat.size(),lambda,gamma);
        }else{
            if(argc <4){
                cerr<<"The model is a kernel and need the train file data as the third argument"<<endl;
		exit(0);
            }
            string train_fileName(argv[3]);
            size_t k = ReadTrainData(train_fileName,data_t,y_t);
            if(kernel_type == "RBF"){
                iss >> sigma;
                ker = new rbfKernel(data_t,sigma);
            }
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
    vector<int> label_map;
    cerr<<"Reading model file"<<endl; 
    svm* sv =   ReadModel(model_file,ker,argc,argv,label_map);


    
    size_t test_size = testData.size();
    if(test_size >0 ){
        ivec y_res(test_size);
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
