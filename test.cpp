#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "usedFun.hpp"

using namespace Eigen;
using namespace std;


typedef vector<double> vec;
typedef vector<int> ivec;
typedef vector<vec> mat;


void ReadData(string fileName,mat& data,ivec & label){
  string line;
  ifstream myfile;
  myfile.open(fileName.c_str(),ifstream::in);
  double tmp;
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
    label.push_back((int)(v.back()));
        v.pop_back();

    data.push_back(v);
  }
  myfile.close();
}


int main(int argc,char ** argv){
    // string  fileName(argv[1]);
    // mat data_t;
    // ivec y_t;
    // ReadData(fileName,data_t,y_t);

    MatrixXd data = MatrixXd::Random(3,4);
    MatrixXd data2(3,4);
    data2 = data;
    data2(2,1) = -3;
    data.col(1).setZero();
    cout<<data<<endl;
    cout<<"__________"<<endl;
    cout<<data2<<endl;

    data2 = data2 + data;
    
cout<<"__________"<<endl;
    cout<<data2<<endl;
    
    
    cout<<data.transpose()* (data.col(2))<<endl;

    unsigned int arr[10];
    for(int i=0;i<5;i++){
        randperm(10,arr);
        for(int j=0;j<10;j++){
            cout<<arr[j]<<" ";
        }
        cout<<endl;
    }
    
    
    return 0;
}

