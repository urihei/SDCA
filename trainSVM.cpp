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
#include "zeroOneRBiasKernel.hpp"
#include "saulZeroKernel.hpp"
#include "saulOneKernel.hpp"
#include "sparseKernelSVM.hpp"
/**
 *
 *
 **/
double findLambda(svm* sv,unsigned int folds,
                  double ll,double lh, unsigned int lamLen,
                  int* label_map,size_t* y,double* validation_err){

  size_t n = sv->getN();
  double step = pow(10,(lh-ll)/(lamLen-1));
  
  size_t train_size = (folds-1.0)/folds * n;
  size_t test_size = n-train_size;
  cerr<<train_size<<" "<<test_size<<endl;

  //setting the validation set.
  sv->setUsedN(train_size);
  sv->samplePrm();
  //prepare for loop
  double best_lambda = -1;
  double best_res = test_size*folds; //worst posibble value - wrong on all test set.
  
  ivec res(test_size);
  ivec res2(train_size); //to remove
  double lambda = pow(10,ll);
  for(size_t lm=0;lm<lamLen;++lm){
    //avoiding to much cerr output
    sv->setVerbose(false);
    cerr<<"LM: "<<lm<<" "<<lambda<<"\t";
    sv->setLambda(lambda);
    unsigned int  err =0; //counter to validation errors.
    //    unsigned int tr_err = 0; //counter to train errors
    for(size_t i=0;i<folds; ++i){
       //we start over from scratch so the alpha will be zero on the validation set.
      sv->init();
   
      double gap; //saving the gap
      //do we use accelerated SGD 
      if(sv->getAccIter() >0){
        gap = sv->learn_acc_SDCA();
      }else{
        gap = sv->learn_SDCA();
      }
      cerr<<i<<": "<<gap<<"\t";

      //testing this lambda
      ivec_iter itb = sv->getPrmArrayBegin()+train_size;
      ivec_iter ite = sv->getPrmArrayEnd();
      sv->classify(itb,ite,res);
      //compare the results.
      size_t j=0;
      for(ivec_iter it =itb; it<ite;++it){
        if(res[j++] != y[*it]){
          err++;
        }
      }
      //begin to remove
      /*
      //testing on the train.
      itb = sv->getPrmArrayBegin();
      ite = sv->getPrmArrayEnd() - test_size;
      sv->classify(itb,ite,res2);
      j=0;
      for(ivec_iter it =itb; it<ite;++it){
        if(res2[j++] != y[*it]){
          tr_err++;
        }
      }
      */
      //end to remove
      //shifting the usedData such we will use
      // another part on the data as validation set
      sv->shiftPrm(test_size);
    }
    //checking if this is the best lambda so far.
    if(err<= best_res){
      best_res = err;
      best_lambda = lambda;
    }
    lambda *= step;//advance the lambda
    //    cerr<<"Train err\t"<<((tr_err+0.0)/(train_size*folds))<<"\t";
    cerr<<"Test Error\t"<<((err+0.0)/(test_size*folds))<<endl;
  }
  *validation_err = best_res/(test_size*folds);
  return best_lambda;
}

void printUsage(char* ex){
  cerr<<ex<<"\t input_file"<<endl;
  
  cerr<<"optional:"<<endl;
  cerr<<"\t"<<"-m model_file"<<" default no saving"<<" The file name the model will be saved to  if not given the model is not saved"<<endl;
  cerr<< "\t"<<"-test test_data"<<"default no testing" <<" A file name of the data to test the classifier on " <<endl;
  cerr<<"\t"<<"-verbose [0|1]"<<" default 1 "<<" Printing information to stderr during run"<<endl;
  cerr<<"\t"<<"-k KernelType[Linear|preCalcKernel|RBF|Poly|ZeroOneL1|ReluL1|ZeroOneL2|ZeroOneR|saulZero|saulOne]"<<"\t default Linear" <<" The kernel type use in svm"<<endl;;
  cerr<<"\t"<<"-lambda value[double]"<<"\t default 1"<<" The l2 regulation parameter"<<endl; ;
  cerr<<"\t"<<"-gamma value[double]"<<"\t default 0.1"<<" The hinge loss smoothing parameter "<<endl;
  cerr<<"\t"<<"-lambda_find [unsigned int]"<<"defult 0"<<"To find the best lambda using 5 fold cross validation"<<endl;
  cerr<<"\t"<<"-iter value[double]"<<"\t default [100|5]"<<" The maximum number of times the whole data is iterated in each SDCA run(if accelerated is used the defualt is 5)"<<endl;
  cerr<<"\t"<<"-iter_acc value[unsigned int]"<<"\t default [0]"<<" The maximum number of times the acclerated algorithm is repeated"<<endl;

  cerr<<"\t"<<"-check_gap value[unsigned int]"<<"\t default [5]"<<" The frequency (the whole data set) the duality  gap is checked in SDCA"<<endl;
  cerr<<"\t"<<"-check_gap_acc value[unsigned int]"<<"\t default [5]"<<" The frequency the stoping condition is checked in acclerated SDCA"<<endl;

  cerr<<"\t"<<"-epsilon value[double]"<<"\t default [1e-3]"<<" The requested accuracy "<<endl;
  cerr<<"\t"<<"-validation  value[unsigned int]"<<"\t default [0]"<<"Cross validation value - the number fo parts to divide the train set. 0 mean do not run this"<<endl;
  cerr<<"\t"<<"-normalize  [0|1]"<<"\t default [0]"<<"Center the data and divide by the max l2 norm"<<endl;
  cerr<<"\t Kernel Option:"<<endl;
  cerr<<"\t\t"<<"-preCalc [0|1]"<<"\t default 0"<<" 1 - For calculate the kernel before optimization, 0 - for on the fly calculation."<<endl;  
  cerr<<"\t\t"<<"-sigma value[double]"<<"\t default 1"<<" Sigam for the RBF kernel"<<endl;
  cerr<<"\t\t"<<"-degree value[double]"<<"\t default 2"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
  cerr<<"\t\t"<<"-c value[double]"<<"\t default 1"<<" Degree for the polynomial kernel (<x,y>+c)^degree"<<endl;
  cerr<<"\t\t"<<"-hidden value[unsigned int]"<<"\t default 20"<<" Number of hidden units in the ZeroOneL2 kernel"<<endl;
  cerr<<"\t\t"<<"-hidden_layer value'-'value'-'...'-'value[unsigned int vector]"<<"\t default [20-10]"<<" Number of hidden units in each layer of the ZeroOneR* kernels"<<endl;
  cerr<<"\t\t"<<"-bias value'-'value'-'...'-'value[double vector]"<<"\t default [1-1-1]"<<" The bias at each layer"<<endl;
  cerr<<"\t\t"<<"-l value[unsigned int]"<<"\t default 0"<<" Number of layers in Saul kernels"<<endl;
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
  unsigned int  lambda_find = 0;
  bool insertLambda = false;
  
  double iter = 100;
  size_t acc_iter = 0;

  unsigned int checkGap = 5;
  unsigned int checkGapAcc = 5;
  unsigned int validate = 0;
  bool normalize = false;
  double epsilon = 1e-3;

  bool preCalc = false;
  double sigma = 1; //RBF
  //  double degree = 2;//Poly
  // double c = 1; //Poly
  unsigned int l = 0;
  // unsigned int hidden = 5;
  ivec hidden_layer {1,20,10};
  vec bias {1.0,1.0,1.0};
    
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
      insertLambda = true;
      rec = true;
    }
    if(strcmp("-gamma",argv[i])==0){
      gamma  = atof(argv[i+1]);
      rec = true;
    }
    if(strcmp("-lambda_find",argv[i])==0){
      lambda_find = atoi(argv[i+1]) ;
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
    if(strcmp("-validation",argv[i])==0){
      validate = atoi(argv[i+1]);;
      rec = true;
    }
    if(strcmp("-normalize",argv[i])==0){
      normalize = argv[i+1][0] == '1';
      rec = true;
    }
    if(strcmp("-sigma",argv[i])==0){
      sigma  = atof(argv[i+1]);
      rec = true;
    }
    if(strcmp("-l",argv[i])==0){
      l  = atoi(argv[i+1]);
      rec = true;
    }
    /*if(strcmp("-degree",argv[i])==0){
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
      */
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
    
    if(! rec){
      cerr<<"option "<<argv[i]<<" is not recognized"<<endl;
      printUsage(argv[0]);
    }
  }
  double* data_t; //matrix to hold the data.
  size_t* y_t; //vec  to hold the data labels mapped to 0-(k-1).
  int* label_map; // vec to hold the map between the labels and the mapped labels.
  size_t n=-1;
  size_t p =-1;
  //reading the data
  size_t k = ReadTrainData(data_file,data_t,y_t,label_map,n,p);// the number of classes
  cerr<<"Finish reading data N:"<<n<<" P: "<<p<<endl;
  //normalize
  double meanVec[p];
  double maxNorm=1;
  if(normalize)
    maxNorm = normalizeData(data_t,meanVec,n,p);
  
  //creating the kernel.
  time_t start_time =time(NULL);
  Kernel* ker= NULL;
  svm* sv;
  if(kernel_type == "Linear"){
    sv = new linearSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
  }else{
    if( kernel_type == "preCalcKernel"){
      cerr<<"Create predefined kernel"<<endl;
      sv = new preKernelSVM(y_t,data_t,k,lambda,gamma,iter*n,acc_iter);
    }else{
      cerr<<"Create zeroOneRBias kernel"<<endl;
      if(kernel_type == "ZeroOneRBias"){
        ker = new zeroOneRBiasKernel(data_t,n,p,hidden_layer,bias);
      }
      if(kernel_type == "RBF"){
        ker = new rbfKernel(data_t,n,p,sigma);
      }
      if(kernel_type == "saulZero"){
        ker = new saulZeroKernel(data_t,n,p,l);
      }
      if(kernel_type == "saulOne"){
        ker = new saulOneKernel(data_t,n,p,l);
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
      if(kernel_type == "LinearKernel"){
        ker = new linearKernel(data_t);
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
      
*/
      if(ker==NULL){
        cerr<<"Unknown kernel: "<<kernel_type<<endl;
        printUsage(argv[0]);
      }
      if(preCalc){
        sv = new preKernelSVM(y_t,ker,k,n,lambda,gamma,iter*n,acc_iter);
      }else{
        sv = new kernelSVM(y_t,ker,k,lambda,gamma,iter*n,acc_iter);
        //        sv = new sparseKernelSVM(y_t,ker,k,n,lambda, gamma,iter*n,acc_iter);
      }
    }
  }
  
  time_t kernel_creation = time(NULL)-start_time;
  cerr<<"Kerenel creating time "<< kernel_creation<<endl;
  cerr<<"Lambda: "<<sv->getLambda()<<endl;
  sv->setVerbose(verbose);
  sv->setCheckGap(checkGap);
  sv->setCheckGapAcc(checkGapAcc);
  sv->setEpsilon(epsilon);

  cerr<<"Finish creating the svm object"<<endl;

  if(lambda_find){
    double val_err;
    double logLambda;
    double le = -8;
    double he = 0;
    unsigned int div = 4;
    double ed = 2.0;
    if(insertLambda){
      logLambda = log10(lambda);
      le = logLambda - ed;
      he = logLambda + ed;
      ed /= 2;
      div = 5;
    } 
    double best_lambda = findLambda(sv,6,le,he,div ,label_map,y_t,&val_err);
    logLambda = log10(best_lambda);
    for(size_t t=0;t<lambda_find;++t){
      best_lambda = findLambda(sv,6,logLambda-ed,logLambda+ed, 5,label_map,y_t,&val_err);
      logLambda = log10(best_lambda);
      ed /= 2;
    }
    cout<<"Lambda:\t"<<best_lambda<<endl;
    cout<<"Validation Err\t"<<val_err<<endl;
    sv->setLambda(best_lambda);
    sv->setUsedN(n);
    sv->setVerbose(verbose);
  }
  if(validate > 1 && !lambda_find){  
    size_t train_size = (validate - 1.0)/validate * n;
    size_t test_size = n-train_size;
    ivec res(n);
    size_t err = 0;
    sv->setVerbose(false);
    sv->setUsedN(train_size);
    sv->samplePrm();
    cerr<<"Starting validation:\t";
    for(size_t i=0;i<validate; ++i){
      cerr<<i<<"\t";
       //we start over from scratch so the alpha will be zero on the validation set.
      sv->init();
   
      //do we use accelerated SGD 
      if(sv->getAccIter() >0){
        sv->learn_acc_SDCA();
      }else{
        sv->learn_SDCA();
      }
      //testing this lambda
      ivec_iter itb = sv->getPrmArrayBegin()+train_size;
      ivec_iter ite = sv->getPrmArrayEnd();
      sv->classify(itb,ite,res);
      //compare the results.
      size_t j=0;
      for(ivec_iter it =itb; it<ite;++it){
        if(res[j++] != y_t[*it]){
          err++;
        }
      }
      //shifting the usedData such we will use
      // another part on the data as validation set
      sv->shiftPrm(test_size);
    }
    cerr<<endl;
    sv->setUsedN(n);
    sv->setVerbose(verbose);
    cout<<"Validation Err\t"<<(err+0.0)/(test_size*validate)<<endl;
  }
  start_time = time(NULL);
  if(acc_iter >0){
    sv->learn_acc_SDCA();
  }else{
    sv->learn_SDCA();
  }
  time_t learning_time = time(NULL)-start_time;
  cout<<"Kernel creation time\t"<<kernel_creation<<"\tLearning time\t"<< learning_time<<endl;
  //
  ivec trErr(n);
  ivec_iter itb = sv->getPrmArrayBegin();
  ivec_iter ite = sv->getPrmArrayEnd();  
  sv->classify(itb,ite,trErr);
  size_t j =0;
  unsigned int err = 0;
  for(ivec_iter it =itb; it<ite;++it){
    cerr<<trErr[j]<<"<->"<<y_t[*it]<<endl;
    if(trErr[j++] != y_t[*it]){
      err++;
    }
  }
  cout<<"Train Err\t"<<(err+0.0)/j<<endl;
  //
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
    double* testData;
    size_t* y_test;
    int* test_label_map;
    size_t n_test =-1;
    size_t p_test = -1;
    ReadTrainData(test_file,testData,y_test,test_label_map,n_test,p_test);
    assert(p_test == p);
    cerr<<"Finsh reading test data"<<endl;
    
    if(normalize)
      normalizeData(testData,meanVec,maxNorm,n_test,p_test);
    
    size_t y_res[n_test];
    sv->classify(testData,y_res,n_test,p_test);
    unsigned int count = 0;
    for(size_t i=0;i<n_test;++i){
      if(label_map[y_res[i]] != test_label_map[y_test[i]])
        count++;
    }
    cout <<count<<"/"<<n_test<<"\t"<<(count+0.0)/n_test<<endl;
    delete testData;
    delete y_test;
    delete test_label_map;
  }

  if(ker != NULL)
    delete ker;

  delete data_t;
  delete y_t;
  delete label_map;

  delete sv;
}

