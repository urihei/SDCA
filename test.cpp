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
    srand( (unsigned)time(NULL) );
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

    ArrayXd a = ArrayXd::Random(10)*1.5;
    ArrayXd b(10);

    cout<<"__________"<<endl;
    cout<<a<<endl;
    cout<<"__________"<<endl;
    b = a.max(0);
    cout<<b<<endl;
    cout<<"__________"<<endl;
    sort(b.data(),b.data()+b.size());
    cout<<b<<endl;
    cout<<"__________"<<endl;
    b.reverseInPlace();
    cout<<b<<endl;
    cout<<"__________"<<endl;
    cumsum(b.data(),b.size(),a.data());
    cout<<a<<endl;
    cout<<"__________"<<endl;
    ArrayXd OneToK(10);
    OneToK.setOnes();
    cumsum(OneToK.data(),10,OneToK.data());
    cout<<OneToK<<endl;
    cout<<"__________"<<endl;
    ArrayXd z(11);
    z.head(10) = (a - (OneToK * b)).min(1);
    z(10) = 1;
    cout<<z<<endl;
    cout<<"__________"<<endl;
    ArrayXd normOne(10);
    normOne = a/(1+(OneToK*0.9));
    cout<<normOne<<endl;
    cout<<"__________"<<endl;
    size_t ind = findFirstBetween(normOne.data(),z.head(10).data(),z.tail(10).data(),10);
    cout<<ind<<endl;
    cout<<"__________"<<endl;
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

