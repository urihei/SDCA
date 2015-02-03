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
#include "linearKernel.hpp"
#include "saulZeroKernel.hpp"
#include "saulOneKernel.hpp"


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


int main(int argc,char ** argv){
    // ivec prm(10);
    // ivec preDefine(10);
    // for(size_t i=0;i<10;++i){
    //     preDefine[i] = i;
    // }
    // randperm(10,prm,preDefine);
    // for(size_t i=0;i<10;++i){
    //     cout<<prm[i]<<" ";
    //     preDefine[i] = prm[i];
    // }
    // cout<<endl;
    // randperm(5,prm,preDefine);
    // for(size_t i=0;i<10;++i){
    //     cout<<prm[i]<<" ";
    //     preDefine[i] = prm[i];
    // }
    // cout<<endl;
    // exit(0);
    vector<unsigned int> v(4);
    unsigned int N = 15;
    unsigned int s1 = 5;
    unsigned int s2 = 3;
    unsigned int i = 2; 
    v[0] = i;
    v[1] = s1-i;
    v[2] = s2-i;
    v[3] = N+i-s1-s2;
    cerr<<N+i-s1-s2<<"-"<<multinomial(v)<<endl;
cerr<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<endl;
    exit(0);
    string  fileName(argv[1]);
    matd data_t;
    ivec y_t;
    size_t k =  ReadData(fileName,data_t,y_t);
    size_t n = y_t.size();
    double lambda = 10/(n+0.0);
    //    polyKernel* ker = new polyKernel(data_t,2,1);
    zeroOneL2Kernel* ker = new zeroOneL2Kernel(data_t,N);
ivec ll {1,10,5}; 
    zeroOneRKernel* ker2 = new zeroOneRKernel(data_t,ll);
    //saulZeroKernel* ker = new saulZeroKernel(data_t,0);
    //saulOneKernel* ker = new saulOneKernel(data_t,0);
    cerr<<"Finish create kernel"<<endl;
    ArrayXd alpha(64);
    for(size_t i=0;i<64;++i){
        double ang = i*M_PI/64;
        alpha(i) = ang;
    }
    VectorXd res(64);
    VectorXd res2(64);
    ker->calc(alpha*OneDpi,res);
cerr<<"finish calc 1"<<endl;
ker2->calc(alpha*OneDpi,res2,2);
    //    alpha = alpha.cos();
//    ker->calcAngle(alpha,5);

    cout<<endl<<"RES:\n"<<res<<endl;
    cout<<endl<<"RES:\n"<<res2<<endl;
    exit(0);

    //    reluL1Kernel* ker = new reluL1Kernel(data_t);
    VectorXd vec(n);
    for(size_t i=0; i<n; ++i){
        ker->dot(i,vec);
        for(size_t j=0; j<n; ++j){
            cout<<vec(j)<<" ";
        }
        cout<<endl;
    }
    exit(0);
    //zeroOneL1Kernel* ker = new zeroOneL1Kernel(data_t);
    //linearKernel* ker = new linearKernel(data_t);
    preKernelSVM svm(y_t,ker,k,lambda,0.1,100*n);
    //kernelSVM svm(y_t,k,ker,lambda,0.0001,100*n);
    //linearSVM svm(y_t,data_t,k,lambda,0.1,100*n);
    mat alpha1(k,n);
    alpha1.setZero();
    mat zW(data_t[0].size(),k);
    zW.setZero();
    mat zAlpha(k,n);
    zAlpha.setZero();
    //  mat zAlpha = MatrixXd::Random(k,n);
    cerr<<"Finish reading data"<<endl;
    time_t start =time(NULL);
    svm.learn_SDCA();//zW);//
    cout<<"time :"<<time(NULL) - start<<endl;
    //return 0;
    //eval
    //classify
    ivec y_res(n);
    svm.classify(data_t,y_res);
    size_t count =0;
    for(size_t i=0;i<n;i++){
        //comapre
        //        cerr<<y_res[i]<<"<->"<<y_t[i]<<endl;
        if(y_res[i] != y_t[i])
            count++;
    }
    cout<<"The Number of train error "<<count<<endl;
    return 0;
    svm.setAccIter(100);
    svm.setIter(5*n);
    start =time(NULL);
    svm.learn_acc_SDCA();
    cout<<"time :"<<time(NULL) - start<<endl;
    svm.classify(data_t,y_res);
    count =0;
    for(size_t i=0;i<n;i++){
        //comapre
        if(y_res[i] != y_t[i])
            count++;
    }
    cout<<"The Number of train error "<<count<<endl;
    /*    kernelSVM svm(y_t,data_t,k,lambda,0.1,100*n,0);

          mat alpha1(k,n);
          alpha1.setZero();
          mat zAlpha(k,n);
          zAlpha.setZero();
          //  mat zAlpha = MatrixXd::Random(k,n);
          cerr<<"Finish reading data"<<endl;
          time_t start =time(NULL);
          svm.learn_SDCA(alpha1,zAlpha);
          cout<<"time :"<<time(NULL) - start<<endl;
          //return 0;
          //eval
          //classify
          ivec y_res(n);
          svm.classify(data_t,y_res);
          size_t count =0;
          for(size_t i=0;i<n;i++){
          //comapre
          //        cerr<<y_res[i]<<"<->"<<y_t[i]<<endl;
          if(y_res[i] != y_t[i])
          count++;
          }
          cout<<"The Number of train error "<<count<<endl;
          mat alpha2(k,n);
          alpha2.setZero();
          svm.setAccIter(100);
          svm.setIter(5*n);
          start =time(NULL);
          svm.learn_acc_SDCA();
          cout<<"time :"<<time(NULL) - start<<endl;
          svm.classify(data_t,y_res);
          count =0;
          for(size_t i=0;i<n;i++){
          //comapre
          if(y_res[i] != y_t[i])
          count++;
          }
          cout<<"The Number of train error "<<count<<endl;*/
    //    srand( (unsigned)time(NULL) );
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
    
    //        delete ker;
    return 0;
}

