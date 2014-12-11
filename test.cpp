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

#include "learnMultiClassSVM.hpp"
using namespace Eigen;
using namespace std;




int ReadData(string fileName,matd& data,ivec & label){
    string line;
    ifstream myfile;
    myfile.open(fileName.c_str(),ifstream::in);
    double tmp;
    map<size_t,int> lMap;
    size_t countLabel = 0;
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
        if(lMap.count(cl) > 0){
            label.push_back(lMap[cl]);
        }else{
            label.push_back(countLabel);
            lMap[cl] = countLabel;
            countLabel++;
        }

        v.pop_back();

        data.push_back(v);
    }
    myfile.close();
    return countLabel;
}


int main(int argc,char ** argv){
  string  fileName(argv[1]);
  matd data_t;
  ivec y_t;
  size_t k =  ReadData(fileName,data_t,y_t);
  size_t n = y_t.size();
  double lambda = 10/(n+0.0);
  learnMultiClassSVM svm(y_t,data_t,k,100*n,0,lambda);
  mat alpha1(k,n);
  alpha1.setZero();
  mat zAlpha(k,n);
  zAlpha.setZero();
  MatrixXd* dataPt = svm.getData();
  MatrixXd data = *dataPt;
  cerr<<"Finish reading data"<<endl;
  time_t start =time(NULL);
  svm.learn_SDCA(alpha1,zAlpha);
  cout<<"time :"<<time(NULL) - start<<endl;
  //eval
  //classify
  MatrixXd ya(k,n);
  ya = 1/(lambda*n) * (alpha1 * data);
  unsigned int  count =0;
  MatrixXf::Index index;
  for(size_t i=0;i<n;i++){
      ya.col(i).maxCoeff(&index);
      //comapre
      if( ((unsigned int)index) != y_t[i])
          count++;
  }
  cout<<"The Number of train error "<<count<<endl;
  
  mat alpha2(k,n);
  alpha2.setZero();
  svm.setAccIter(100);
  svm.setIter(5*n);
  start =time(NULL);
  svm.acc_learn_SDCA(alpha2);
    cout<<"time :"<<time(NULL) - start<<endl;
  ya = 1/(lambda*n) * (alpha2 * data);
  count =0;
  for(size_t i=0;i<n;i++){
      ya.col(i).maxCoeff(&index);
      //comapre
      if( ((unsigned int)index) != y_t[i])
          count++;
  }
  cout<<"The Number of train error "<<count<<endl;
  srand( (unsigned)time(NULL) );
    // MatrixXd data = MatrixXd::Random(3,4);
    // MatrixXd data2(3,4);
    // data2 = data;
    // data2(2,1) = -3;
    // data.col(1).setZero();
    // cout<<data<<endl;
    // cout<<"__________"<<endl;
    // cout<<data2<<endl;

    // data2 = data2 + data;
    
    // cout<<"__________"<<endl;
    // cout<<data2<<endl;
    
    
    // cout<<data.transpose()* (data.col(2))<<endl;

    // ArrayXd a = ArrayXd::Random(10)*1.5;
    // ArrayXd b(10);

    // cout<<"__________"<<endl;
    // cout<<a<<endl;
    // cout<<"__________"<<endl;
    // b = a.max(0);
    // cout<<b<<endl;
    // cout<<"__________"<<endl;
    // sort(b.data(),b.data()+b.size());
    // cout<<b<<endl;
    // cout<<"__________"<<endl;
    // b.reverseInPlace();
    // cout<<b<<endl;
    // cout<<"__________"<<endl;
    // cumsum(b.data(),b.size(),a.data());
    // cout<<a<<endl;
    // cout<<"__________"<<endl;
    // ArrayXd OneToK(10);
    // OneToK.setOnes();
    // cumsum(OneToK.data(),10,OneToK.data());
    // cout<<OneToK<<endl;
    // cout<<"__________"<<endl;
    // ArrayXd z(10);
    // z = (a - (OneToK * b)).min(1);
    // cout<<z<<endl;
    // cout<<"__________"<<endl;
    // ArrayXd normOne(10);
    // normOne = a/(1+(OneToK*0.9));
    // cout<<normOne<<endl;
    // cout<<"__________"<<endl;
    // size_t ind = findFirstBetween(normOne.data(),z.data(),10);
    // cout<<ind<<endl;
    // cout<<"__________"<<endl;
    // unsigned int arr[10];
    // for(int i=0;i<5;i++){
    //     randperm(10,arr);
    //     for(int j=0;j<10;j++){
    //         cout<<arr[j]<<" ";
    //     }
    //     cout<<endl;
    // }
    
    
    return 0;
}

