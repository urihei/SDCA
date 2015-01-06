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




int ReadData(string fileName,matd& data,ivec & label);

void printUsage(char* ex){
    cerr<<ex<<"\t input_file"<<endl;

    cerr<<"optional:"<<endl;
    cerr<<"\t"<<"-m model_file"<<" default no saving"<<" The file name the model will be saved to  if not given the model is not saved"<<endl;
    cerr<< "\t"<<"-test test_data"<<"default no testing" <<" A file name of the data to test the classifier on " <<endl;
    cerr<<"\t"<<"-verbose [0|1]"<<" default 1 "<<" Printing information to stderr during run"<<endl;
    cerr<<"\t"<<"-k KernelType[Linear|preCalcKernel|RBF|Poly|ZeroOneL1|ReluL1]"<<"\t default Linear" <<" The kernel type use in svm"<<endl;;
    cerr<<"\t"<<"-lambda value[double]"<<"\t default 1"<<" The l2 regulation parameter"<<endl; ;
    cerr<<"\t"<<"-gamma value[double]"<<"\t default 0.1"<<" The hinge loss smoothing parameter "<<endl;
    cerr<<"\t"<<"-iter value[double]"<<"\t default [100|5]"<<" The maximum number of times the whole data is iterated in each SDCA run(if accelerated is used the defualt is 5)"<<endl;
    cerr<<"\t"<<"-iter_acc value[unsigned int]"<<"\t default [0]"<<" The maximum number of times the acclerated algorithm is repeated"<<endl;

    cerr<<"\t"<<"-check_gap value[unsigned int]"<<"\t default [5]"<<" The frequency (the whole data set) the duality  gap is checked in SDCA"<<endl;
    cerr<<"\t"<<"-check_gap_acc value[unsigned int]"<<"\t default [5]"<<" The frequency the stoping condition is checked in acclerated SDCA"<<endl;

    cerr<<"\t"<<"-epsilon value[double]"<<"\t default [1e-3]"<<" The requested accuracy "<<endl;
    
    cerr<<"\t Kernel Option:"<<endl;
    cerr<<"\t\t"<<"-sigma value[double]"<<"\t default 1"<<" Sigam for the RBF kernel"<<endl;
    cerr<<"\t\t"<<"-degree value[double]"<<"\t default 2"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
   cerr<<"\t\t"<<"-c value[double]"<<"\t default 1"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
   
    exit(1);
}

int main(int argc,char ** argv){
    if(argc < 2){
        printUsage(argv[0]);
    }
    string data_file(argv[1]);

    string model_file = "";
    string test_file = "";
    bool verbose = true;
    
    string kernel_type = "Linear";
    double lambda = 1;
    double gamma = 0.1;

    double iter = 100;
    size_t acc_iter = 0;

    unsigned int checkGap = 5;
    unsigned int checkGapAcc = 5;

    double epsilon = 1e-3;
    
    double sigma = 1; //RBF
    double degree = 2;//Poly
    double c = 1; //Poly
    
    for(int i=2;i<argc; i+= 2){
        bool rec = false;
        if(strcmp("-m",argv[i])==0){
            model_file = string(argv[i+1]);
            rec = true;
        }
        if(strcmp("-test",argv[i])==0){
            test_file = string(argv[i+1]);
            rec = true;
        }
        if(strcmp("-verbose",argv[i])==0){
            verbose = argv[i+1][0] == '1';
            rec = true;
        }
        if(strcmp("-k",argv[i])==0){
            kernel_type  = string(argv[i+1]);
            rec = true;
        }
        if(strcmp("-lambda",argv[i])==0){
            lambda  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-gamma",argv[i])==0){
            gamma  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-iter",argv[i])==0){
            iter  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-iter_acc",argv[i])==0){
            acc_iter  = atoi(argv[i+1]);
            rec = true;
        }
        if(strcmp("-check_gap",argv[i])==0){
            checkGap  = atoi(argv[i+1]);
            rec = true;
        }
        if(strcmp("-check_gap_acc",argv[i])==0){
            checkGapAcc  = atoi(argv[i+1]);
            rec = true;
        }
        if(strcmp("-epsilon",argv[i])==0){
            epsilon  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-sigma",argv[i])==0){
            sigma  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-degree",argv[i])==0){
            degree  = atof(argv[i+1]);
            rec = true;
        }
        if(strcmp("-c",argv[i])==0){
            c  = atof(argv[i+1]);
            rec = true;
        }
        if(! rec){
            cerr<<"option "<<argv[i]<<" is not recognized"<<endl;
            printUsage(argv[0]);
        }
    }
    
    matd data_t;
    ivec y_t;
    size_t k =  ReadData(data_file,data_t,y_t);
    size_t n = y_t.size();
    Kernel* ker= NULL;
    svm* sv;
    if(kernel_type == "Linear"){
        sv = new linearSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
    }else{
        if( kernel_type == "preCalcKernel"){
            sv = new preKernelSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
        }else{

            if(kernel_type == "RBF"){
                ker = new rbfKernel(data_t,sigma);
            }
            if(kernel_type == "Poly"){
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
            sv = new kernelSVM(y_t,k,ker,lambda,gamma,iter*n,acc_iter);
        }
    }
    sv->setVerbose(verbose);
    sv->setCheckGap(checkGap);
    sv->setCheckGapAcc(checkGapAcc);
    sv->setEpsilon(epsilon);



    
    if(acc_iter >0){
        sv->learn_acc_SDCA();
    }else{
        sv->learn_SDCA();
    }


    if(model_file != ""){
        sv->saveModel(model_file);
    }
    if(test_file != ""){
        matd testData;
        ivec y_test;
        ReadData(test_file,testData,y_test);
        size_t test_size = y_test.size();
        ivec y_res(test_size);
        sv->classify(testData,y_res);
        unsigned int count = 0;
        for(size_t i=0;i<test_size;++i){
            if(y_res[i] != y_test[i])
                count++;
        }
        cout <<count<<"/"<<test_size<<"\t"<<count/test_size<<endl;
    }

    if(ker != NULL)
        delete ker;

    delete sv;
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
