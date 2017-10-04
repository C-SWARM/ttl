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
#include <ttl/ttl.h>

using namespace ttl;
using namespace ttl::expressions;
using std::tuple;
using std::is_same;

// Declare i, j, k index types.
using i = Index<'i'>;
using j = Index<'j'>;
using k = Index<'k'>;

// Build some base index lists based on i,j,k
using l_empty = tuple<         >;
using     l_i = tuple< i       >;
using     l_j = tuple< j       >;
using     l_k = tuple< k       >;
using    l_ij = tuple< i, j    >;
using    l_ji = tuple< j, i    >;
using    l_ik = tuple< i, k    >;
using    l_jk = tuple< j, k    >;
using   l_ijk = tuple< i, j, k >;
using   l_kij = tuple< k, i, j >;

// Test the non-integral-type filter
using   l_int_i = tuple<int, i>;
using   l_i_int = tuple<i, int>;
using l_i_int_k = tuple<i, int, k>;

static_assert( equivalent<tuple<>, non_integral<tuple<int>>>::value, "failed");

static_assert( equivalent<l_i, non_integral<l_int_i>>::value, "failed");
static_assert( equivalent<l_i, non_integral<l_i_int>>::value, "failed");
static_assert( equivalent<l_ik, non_integral<l_i_int_k>>::value, "failed");

// Test equivalency
static_assert(  is_same< l_i,     l_i >::value, "failed");
static_assert( !is_same< l_i,     l_j >::value, "failed");
static_assert(  is_same< l_ij,   l_ij >::value, "failed");
static_assert( !is_same< l_ij,   l_ji >::value, "failed");
static_assert(  is_same< l_ijk, l_ijk >::value, "failed");
static_assert( !is_same< l_ijk, l_kij >::value, "failed");

static_assert(  equivalent< l_i,     l_i >::value, "failed");
static_assert( !equivalent< l_i,     l_j >::value, "failed");
static_assert(  equivalent< l_ij,   l_ij >::value, "failed");
static_assert(  equivalent< l_ij,   l_ji >::value, "failed");
static_assert(  equivalent< l_ijk, l_ijk >::value, "failed");
static_assert(  equivalent< l_ijk, l_kij >::value, "failed");

// Create some joined lists
using   j_ii = concat< l_i,  l_i  >;
using   j_ij = concat< l_i,  l_j  >;
using   j_ji = concat< l_j,  l_i  >;
using  j_ijk = concat< j_ij, l_k  >;
using  j_kij = concat< l_k,  j_ij >;
using j_ijij = concat< j_ij, j_ij >;
using j_jiij = concat< j_ji, j_ij >;

static_assert (index_of<i, j_ii>::value == 0, "failed\n");
static_assert (index_of<i, j_ij>::value == 0, "failed\n");
static_assert (index_of<j, j_ij>::value == 1, "failed\n");
static_assert (index_of<k, j_ij>::value == 2, "failed\n");
static_assert (index_of<i, j_ijk>::value == 0, "failed\n");
static_assert (index_of<j, j_ijk>::value == 1, "failed\n");
static_assert (index_of<k, j_ijk>::value == 2, "failed\n");

// Test equivalency
static_assert(  is_same< l_ij,  j_ij  >::value, "failed");
static_assert( !is_same< l_ji,  j_ij  >::value, "failed");
static_assert(  is_same< l_ijk, j_ijk >::value, "failed");
static_assert( !is_same< l_ijk, j_kij >::value, "failed");
static_assert(  is_same<j_ijij, tuple<i,j,i,j>>::value, "failed");
static_assert(  is_same<j_jiij, tuple<j,i,i,j>>::value, "failed");

static_assert(  equivalent< l_ij,  j_ij  >::value, "failed");
static_assert(  equivalent< l_ji,  j_ij  >::value, "failed");
static_assert(  equivalent< l_ijk, j_ijk >::value, "failed");
static_assert(  equivalent< l_ijk, j_kij >::value, "failed");

// Create some intersections
using i_empty1 = set_and< l_i,   l_j   >;
using     i_i1 = set_and< l_i,   l_i   >;
using     i_i2 = set_and< l_i,   j_ij  >;
using     i_i3 = set_and< l_ji,  l_ik  >;
using    i_ij1 = set_and< j_ij,  l_ij  >;
using    i_ij2 = set_and< l_ij,  j_ji  >;
using    i_ij3 = set_and< l_ij,  j_ijk >;
using    i_ij4 = set_and< l_ijk, j_ji  >;

// Check intersections
static_assert( is_same< i_empty1, l_empty >::value, "failed");
static_assert( is_same< i_i1,   l_i >::value, "failed");
static_assert( is_same< i_i2,   l_i >::value, "failed");
static_assert( is_same< i_ij1, l_ij >::value, "failed");
static_assert( is_same< i_ij2, l_ij >::value, "failed");
static_assert( is_same< i_ij3, l_ij >::value, "failed");
static_assert( is_same< i_ij4, l_ij >::value, "failed");

// Create some symmetric differences
using x_empty1 = set_xor< l_i,   l_i   >;
using x_empty2 = set_xor< l_ij,  l_ji  >;
using x_empty3 = set_xor< l_ijk, j_kij >;
using    x_ij1 = set_xor< l_i,   l_j   >;
using    x_jk1 = set_xor< l_ji,  l_ik  >;
using    x_jk2 = set_xor< l_ijk, l_i   >;

// Check symmetric differences
static_assert( is_same< x_empty1, l_empty >::value, "failed");
static_assert( is_same< x_empty2, l_empty >::value, "failed");
static_assert( is_same< x_empty3, l_empty >::value, "failed");

static_assert( is_same< x_ij1, l_ij >::value, "failed");
static_assert( is_same< x_jk1, l_jk >::value, "failed");
static_assert( is_same< x_jk2, l_jk >::value, "failed");

int main() {
  return 0;
}
