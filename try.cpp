#include "usedFun.hpp"
#include "linearSVM.hpp"
#include "preKernelSVM.hpp"

int main(){
  double* data; //mtarix to hold the data.
  size_t* y; //vec  to hold the data labels mapped to 0-(k-1).
  int* label_map; // vec to hold the map between the labels and the mapped labels.
  size_t n=-1;
  size_t p =-1;
  //reading the data
  size_t k =  ReadTrainData("/tmp/ker.txt",data,y,label_map,n,p);
  cerr<<"P data"<<data<<endl;
  preKernelSVM* sv = new preKernelSVM(y,data,k,n,0.001,0.01,500,50);
  cerr<<"start train"<<endl;
  sv->learn_acc_SDCA();
  cerr<<"End train"<<endl;
  
  double* test_data; //mtarix to hold the data.
  size_t* test_y; //vec  to hold the data labels mapped to 0-(k-1).
  int* test_label_map; // vec to hold the map between the labels and the mapped labels.
  size_t test_n=-1;
  size_t test_p =-1;
  size_t test_k =  ReadTrainData("/tmp/kerT.txt",test_data,test_y,test_label_map,test_n,test_p);
  cerr<<test_data[0]<<" N "<<test_n<<" P "<<test_p<<endl;
  size_t y_res[test_n];
  cerr<<"P test data"<<test_data<<endl;
  cerr<<test_n<<endl;
  sv->classify(test_data,y_res,test_n,test_p);
  cerr<<"end classify "<<test_n<<endl;
  unsigned int count = 0;
  for(size_t i=0;i<test_n;++i){
    cerr<<y_res[i]<<"<->"<<test_y[i]<<endl;
    if(label_map[y_res[i]] != test_label_map[test_y[i]])
      count++;
  }
  cout <<count<<"/"<<test_n<<"\t"<<(count+0.0)/test_n<<endl;
  return 0;
}
/*
   double arr[6];
  arr[0] = 1;
  arr[1] = 2;
  arr[2] = 3;
  arr[3] = 4;
  arr[4] = 5;
  arr[5] = 6;
  for(size_t i = 0; i< 6; ++i)
    cout<< arr[i]<<",";
  cout <<endl;
  cout<<"##########################"<<endl;
  Map<MatrixXd> m(NULL,2,2);
  
  new (&m) Map<MatrixXd>(arr,2,3);
  
  for(size_t i = 0; i< 2; ++i){
    for(size_t j=0; j<3;++j)
      cout<< m(i,j)<<",";
    cout<<endl;
  }
  cout<<"##########################"<<endl;
  arr[0]=-100;
  for(size_t i = 0; i< 2; ++i){
    for(size_t j=0; j<3;++j)
      cout<< m(i,j)<<",";
    cout<<endl;
  }
  
  double* data; //mtarix to hold the data.
  size_t* y; //vec  to hold the data labels mapped to 0-(k-1).
  int* label_map; // vec to hold the map between the labels and the mapped labels.
  size_t n=-1;
  size_t p =-1;
  //reading the data
  size_t k =  ReadTrainData("/tmp/data.txt",data,y,label_map,n,p);
  cerr<<"N:"<<n<<" P :"<<p<<endl;
  for(size_t i=0;i<n;i++){
    for(size_t j=0; j<p; ++j){
      cout<< data[i*p+j]<<", ";
    }
    cout<<endl;
  }
  cout<<"##########################"<<endl;
  //Map<MatrixXd> test(NULL,2,2);
  Map<Matrix<double,Dynamic,Dynamic,RowMajor>> test(NULL,1,1);
  new (&test) Map<MatrixXd>(data,n,p);
  for(size_t i = 0; i< n; ++i){
    for(size_t j=0; j<p;++j)
      cout<< test(i,j)<<",";
    cout<<endl;
  }
  cout<<"##########################"<<endl;
  double meanVec [p];
  double var = normalizeData(data,meanVec,n,p);
  for(size_t i = 0; i < n; ++i){
    for(size_t j = 0; j<p; ++j){
      cout << test(i,j)<<",";
    }
    cout<<endl;
  }
  cout<<"##########################"<<endl;
  for(size_t i = 0; i < n; ++i){
    cerr<<y[i]<<",";
  }
*/
