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
#include "kernelSVM.hpp"
#include "linearSVM.hpp"
#include "preKernelSVM.hpp"
#include "rbfKernel.hpp"
#include "polyKernel.hpp"
#include "zeroOneL1Kernel.hpp"
#include "reluL1Kernel.hpp"
#include "zeroOneL2Kernel.hpp"
//#include "linearKernel.hpp"




void printUsage(char* ex){
    cerr<<ex<<"\t input_file"<<endl;

    cerr<<"optional:"<<endl;
    cerr<<"\t"<<"-m model_file"<<" default no saving"<<" The file name the model will be saved to  if not given the model is not saved"<<endl;
    cerr<< "\t"<<"-test test_data"<<"default no testing" <<" A file name of the data to test the classifier on " <<endl;
    cerr<<"\t"<<"-verbose [0|1]"<<" default 1 "<<" Printing information to stderr during run"<<endl;
    cerr<<"\t"<<"-k KernelType[Linear|preCalcKernel|RBF|Poly|ZeroOneL1|ReluL1|ZeroOneL2]"<<"\t default Linear" <<" The kernel type use in svm"<<endl;;
    cerr<<"\t"<<"-lambda value[double]"<<"\t default 1"<<" The l2 regulation parameter"<<endl; ;
    cerr<<"\t"<<"-gamma value[double]"<<"\t default 0.1"<<" The hinge loss smoothing parameter "<<endl;
    cerr<<"\t"<<"-lambda_find [0|1]"<<"defult 0"<<"To find the best lambda using 5 fold cross validation"<<endl;
    cerr<<"\t"<<"-iter value[double]"<<"\t default [100|5]"<<" The maximum number of times the whole data is iterated in each SDCA run(if accelerated is used the defualt is 5)"<<endl;
    cerr<<"\t"<<"-iter_acc value[unsigned int]"<<"\t default [0]"<<" The maximum number of times the acclerated algorithm is repeated"<<endl;

    cerr<<"\t"<<"-check_gap value[unsigned int]"<<"\t default [5]"<<" The frequency (the whole data set) the duality  gap is checked in SDCA"<<endl;
    cerr<<"\t"<<"-check_gap_acc value[unsigned int]"<<"\t default [5]"<<" The frequency the stoping condition is checked in acclerated SDCA"<<endl;

    cerr<<"\t"<<"-epsilon value[double]"<<"\t default [1e-3]"<<" The requested accuracy "<<endl;
    
    cerr<<"\t Kernel Option:"<<endl;
    cerr<<"\t\t"<<"-preCalc [0|1]"<<"\t default 0"<<" 1 - For calculate the kernel before optimization, 0 - for on the fly calculation."<<endl;  
    cerr<<"\t\t"<<"-sigma value[double]"<<"\t default 1"<<" Sigam for the RBF kernel"<<endl;
    cerr<<"\t\t"<<"-degree value[double]"<<"\t default 2"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
    cerr<<"\t\t"<<"-c value[double]"<<"\t default 1"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
    cerr<<"\t\t"<<"-hidden value[unsigned int]"<<"\t default 20"<<" Number of hidden units in the kernel"<<endl; 
    exit(1);
}

int main(int argc,char ** argv){
    if(argc < 2 || argc % 2 != 0){
        printUsage(argv[0]);
    }
    string data_file(argv[1]);

    string model_file = "";
    string test_file = "";
    bool verbose = true;
    
    string kernel_type = "Linear";
    double lambda = 1;
    double gamma = 0.1;
    bool lambda_find = false;
    
    double iter = 100;
    size_t acc_iter = 0;

    unsigned int checkGap = 5;
    unsigned int checkGapAcc = 5;

    double epsilon = 1e-3;

    bool preCalc = false;
    double sigma = 1; //RBF
    double degree = 2;//Poly
    double c = 1; //Poly
    unsigned int hidden = 5;
    
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
	if(strcmp("-lambda_find",argv[i])==0){
	  lambda_find = argv[i+1][0] == '1';
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
        if(strcmp("-preCalc",argv[i])==0){
            preCalc = argv[i+1][0] == '1';
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
        if(strcmp("-hidden",argv[i])==0){
            hidden  = atoi(argv[i+1]);
            rec = true;
        }
        if(! rec){
            cerr<<"option "<<argv[i]<<" is not recognized"<<endl;
            printUsage(argv[0]);
        }
    }
    
    matd data_t;
    ivec y_t;
    vector<int> label_map;
    ReadTrainData(data_file,data_t,y_t,label_map);
    size_t k =  label_map.size();
    cerr<<"Finish reading data"<<endl;
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
            if(kernel_type == "ZeroOneL2"){
                ker = new zeroOneL2Kernel(data_t,hidden);
            }
            if(ker==NULL){
                cerr<<"Unknown kernel: "<<kernel_type<<endl;
                printUsage(argv[0]);
            }
            if(preCalc){
                sv = new preKernelSVM(y_t,ker,k,lambda,gamma,iter*n,acc_iter);
            }else{
                sv = new kernelSVM(y_t,ker,k,lambda,gamma,iter*n,acc_iter);
            }
        }
    }
    sv->setVerbose(verbose);
    sv->setCheckGap(checkGap);
    sv->setCheckGapAcc(checkGapAcc);
    sv->setEpsilon(epsilon);

    cerr<<"Finish creating the svm object"<<endl;

    if(lambda_find){
        unsigned int folds = 5;
        double lambda_val[] = {1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e2};
        cerr<<"Finding lambda"<<endl;
        size_t train_size = (folds-1.0)/folds * n;
	size_t test_size = n-train_size;
	cerr<<train_size<<" "<<test_size<<endl;
        sv->setUsedN(train_size);
        sv->samplePrm();
        double best_lambda = -1;
        double best_res = test_size*folds;
	matd testSet;
	testSet.resize(test_size);
	ivec res(test_size);
        for(size_t lm=0;lm<10;++lm){
	  sv->setVerbose(false);
	  cerr<<"LM: "<<lm<<" "<<lambda_val[lm]<<"\t";
            sv->setLambda(lambda_val[lm]);
	    unsigned int  err =0;
            for(size_t i=0;i<folds; ++i){
                if(acc_iter >0){
                    sv->learn_acc_SDCA();
                }else{
                    sv->learn_SDCA();
                }
		//		cerr<<"geting perm array"<<endl;
                ivec_iter itb = sv->getPrmArrayBegin()+train_size;
		ivec_iter ite = sv->getPrmArrayEnd();
		sv->classify(itb,ite,res);
		size_t j=0;
		for(ivec_iter it =itb; it<ite;++it){
		  //	  cerr<<*it<<" "<<res[j]<<" "<<y_t[*it]<<endl;
		  if(label_map[res[j++]] != label_map[y_t[*it]])
		    err++;
		}
		sv->shiftPrm(test_size);
            }
	    if(err<= best_res){
	      best_res = err;
	      best_lambda = lambda_val[lm];
	    }
	    cerr<<err<<endl;
        }
	cout<<"Lambda:\t"<<best_lambda<<endl;
	sv->setLambda(best_lambda);
	sv->setVerbose(verbose);
    }
    
    if(acc_iter >0){
        sv->learn_acc_SDCA();
    }else{
        sv->learn_SDCA();
    }
    
    if(model_file != ""){
        FILE* pFile = fopen(model_file.c_str(),"w");
        for(size_t i=0;i<k;++i){
            fprintf(pFile,"%zu\t%i\t",i,label_map[i]);
        }
        fprintf(pFile,"\n");
        sv->saveModel(pFile);
        fclose(pFile);
    }
    
    if(test_file != ""){
        matd testData;
        ivec y_test;
        vector<int> test_label_map;
        ReadTrainData(test_file,testData,y_test,test_label_map);
        cerr<<"Finsh reading test data"<<endl;
        size_t test_size = y_test.size();
        ivec y_res(test_size);
        sv->classify(testData,y_res);
        unsigned int count = 0;
        for(size_t i=0;i<test_size;++i){
            if(label_map[y_res[i]] != test_label_map[y_test[i]])
                count++;
        }
        cout <<count<<"/"<<test_size<<"\t"<<(count+0.0)/test_size<<endl;
    }

    if(ker != NULL)
        delete ker;

    delete sv;
}

