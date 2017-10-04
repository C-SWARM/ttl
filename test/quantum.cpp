// -------------------------------------------------------------------*- C++ -*-
// Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
// University of Notre Dame
// Indiana University
// University of Washington
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------
/// A simple test of a quantum logic gate.
///
/// Particles are initialized on eigenvectors, One is rotated using a Hadamard
/// matrix, then they are entangled using a CNOT gate.  Finally, a measurement
/// of one particle, defines the value of the other one.
///
/// This code uses inner and outer products of various sized tensors. Plus
/// Tensor templates & functions intaking and returning Tensors.
///
/// AUTHOR: Ivan Viti
/// -----------------------------------------------------------------------------
#include <type_traits>
#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <mkl.h>
#include <ctime>
#include <ttl/ttl.h>
#include <cstdlib>
#include <time.h>

#include <cmath>
using namespace std;

template<int R,int D = 3,class S = double>
using Tensor = ttl::Tensor<R,D,S>;

constexpr ttl::Index<'i'> i;
constexpr ttl::Index<'j'> j;
constexpr ttl::Index<'k'> k;
constexpr ttl::Index<'l'> l;

template <int R, int D, typename S>
void manualPrint(ttl::Tensor<R,D,S>& A){
    int i,j;
    for (i = 0; i < pow(D,R); i++){
        std::cout << A.get(i) << " ";
        if ((i + 1) % D == 0){
            std::cout << "\n";
        }
    }
}

template <int R, int D, typename S>
ttl::Tensor<2,2,double>  collapse(ttl::Tensor<R,D,S>& A){
        ttl::Tensor<2,2,double> B = {};
    int num = pow(D,R);
    double sums[num];
    int i,j;
    srand(time(NULL));
    double obs = (double) rand()/RAND_MAX;
    sums[0] = 0;
    int picked = 0;
    for (i = 1; i < pow(D,R) ; i++ ){
        sums[i] = sums[i - 1] + A.get(i)*A.get(i);  //squared sum reduction
    }

    for(i = 0; i < pow(D,R); i++) {
        if(obs < sums[i] && picked==0) {
            picked = i;             //tower monte carlo method
        }
    }

        for(i = 0; i < pow(D,R); i++) {
        if(i == picked) {
            B.get(i) = 1;
        }
        else {
            B.get(i) = 0;
        }
    }

return B;
}


void runGate(){
    int idx;
        ttl::Index<'i'> i;
        ttl::Index<'j'> j;

        ttl::Tensor<1,2,double> a = {0,1};
        ttl::Tensor<1,2,double> b = {1,0};
        ttl::Tensor<1,2,double> Hb = {};

    ttl::Tensor<1,4,double> HbaConvert = {};
    ttl::Tensor<1,4,double> entangled = {};

    ttl::Tensor<2,2,double> H = {1/sqrt(2),1/sqrt(2),1/sqrt(2),-1/sqrt(2)};
    ttl::Tensor<2,2,double> Hba = {};
    ttl::Tensor<2,2,double> eConvert = {};
    ttl::Tensor<2,2,double> collapsed = {};

    ttl::Tensor<2,4,double> CNOT = {1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,1,0};


    std::cout<<"\nvector a = \n";
    manualPrint(a);
        std::cout<<"\nvector b = \n";
        manualPrint(b);
    std::cout<<"\nwe then rotate vector b using Hadamard gate:\n";
        manualPrint(H);
    std::cout<<"\nto get:\n";
    Hb = H(i,j)*b(j);
    manualPrint(Hb);
    std::cout<<"\nthis gets correlated (tensor product) with vector a to produce:\n";
    Hba = Hb(i)*a(j);
    manualPrint(Hba);
    std::cout<<"\nwe then entangle these two vectors using a CNOT gate:\n";
    for(idx = 0; idx < 4; idx++) {
        HbaConvert.get(idx) = Hba.get(idx);
    }
    manualPrint(CNOT);
    std::cout<<"\nto get:\n";
    entangled = CNOT(i,j)*HbaConvert(j);
    for(idx = 0; idx < 4; idx++) {
                eConvert.get(idx) = entangled.get(idx);
        }
    manualPrint(eConvert);
    std::cout<<"\nnow, if we measure b as :\n"<<endl;
    collapsed = collapse(eConvert);
    b.get(0) = collapsed.get(0) + collapsed.get(1);
    b.get(1) = collapsed.get(2) + collapsed.get(3);
    manualPrint(b);
        std::cout<<"\nthen we see a as :\n"<<endl;
    a.get(0) = collapsed.get(0) + collapsed.get(2);
    a.get(1) = collapsed.get(1) + collapsed.get(3);
    manualPrint(a);

}


int main(){
runGate();


  return 0;
}

