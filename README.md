A Tensor Template Library

# Configuration

TTL uses cmake as a build framework, and gtest as a unit test
infrastructure. TTL is designed to fetch gtest on its own during
confguration. TTL does not use any unusual cmake variables. A normal
configuration sequence looks something like:

```
$ git clone ttl
$ mkdir ttl-build
$ cd ttl-build
$ cmake ../ttl -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/ttl-install
$ make -j
$ make test
$ make install
$ export C_INCLUDE_PATH="$HOME/ttl-install/include:$C_INCLUDE_PATH"
$ export CPLUS_INCLUDE_PATH="$HOME/ttl-install/include:$CPLUS_INCLUDE_PATH"
```

## BLAS/LAPACK

TTL uses LAPACK to perform inversions and solve operations on higher-dimensional
structures. It uses the FindLAPACK cmake command to set up the LAPACK
dependencies, but this can sometimes to weird things that causes the build to
fail. Some examples that I've seen.

* Finds a BLAS from a different distribution than the LAPACK.
* Finds BLAS and LAPACK correctly but doesn't set up the link correclty.
* Finds BLAS and LAPACK but can't find lapacke.h.

If you are having problems with LAPACK it can pay to specifically request a
LAPACK implementation during configuration by setting the `BLA_VENDOR`
environment variable. Currrent vendors are listed at
https://github.com/Kitware/CMake/blob/master/Modules/FindBLAS.cmake#L35.

For the modern Intel MKL I typically use `Intel10_64lp_seq` (we want sequential
implementations, not parallel implementations.

```
$ cmake <path-to-ttl> -DBLA_VENDOR=Intel10_64lp_seq
```

## Mac OS X

The mac environment provides a lapack implementation in the Accelerate
framework, but it doesn't provide the lapacke bindings that we need. If you want
to use TTL on a mac, then you can either use Intel's MKL if you have it, or you
can install a secondary implementation of lapack. I usually use the homebrew
installation.

If you are using an alternative lapack installation you need to make sure

1. That TTL's cmake can find it.
2. That TTL choses it over the built in Accelerate framework.

You do this on the cmake configure line.

```
$ export LAPACKROOT=<path-to-alternative-lapack>
$ CXXFLAGS=-I$LAPACKROOT/include cmake <path-to-ttl> -DBLA_VENDOR=Generic -DCMAKE_LIBRARY_PATH=$LAPACKROOT/lib <other -D>
```

# User Guide

TTL defines two primary user-defined templates, the Index and the Tensor
templates. An Index is simply a template that converts 8-bit integers into
unique types. Tensors define and shape data storage. Indices appear in Tensor
binding operations and implement symbolic matching between tensor dimensions
within expressions.

## Index

The Index class template simply creates unique types for each char ID that it is
parameterized with. Index values that have the same class are assumed to be the
same index. The resulting type can be manipulated by the TTL internal
infrastructure at compile time. This allows TTL to perform various set
operations on indices in order to perform index matching and code generation for
expressions.

Source-level indices only occur in constant, compile-time contexts and thus
it is common to see them declared as constexpr, const, or both.

```
Tensor<2,2,int> A, B = {...}, C = {...};

constexpr const Index<'i'> i;
constexpr const Index<'j'> j;
constexpr const Index<'k'> k;

// matrix-matrix multiply in TTL (contracts `j`)
C(i,k) = A(i,j) * B(j,k);

// a weirder operation in TTL (contracts `j` again, but different "slot"
C(i,k) = A(j,i) * B(k,j);
```

## Tensor

The Tensor generally defines a square multidimensional, row-major array
structure of statically known sizes. From a storage perspective a basic Tensor
and a statically-sized multidimensional array are effectively the same thing, as
seen in the following example.

```
Tensor<4,3,double> A;
A[1][2][0][1] = 1;
double a[3][3][3][3];
a[1][2][0][1] = 1;
```

Tensors can be initialized elementwise, using initializer lists, and through the
use of the expected copy/move operators.

```
Tensor<2,3,int> A = { 1, 2, 3,
                      4, 5, 6,
                      7, 8, 9 };

Tensor<2,3,double> Uninitialized;
for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 3; ++j) {
    Uninitialized[i][j] = 1.0;
  }
}

Tensor<4,4,double> Zero = {};
```

The major difference between a basic Tensor and a statically sized, square
array, is that a Tensor can be "bound" using its family of operator()(...)
implementations. A bound Tensor has access to the entire suite of Tensor
expressions defined in TTL. A Tensor can be bound either using integers in the
range [0,Dimension) or using Index<> types (@see ttl/Index.h) or both. Binding a
Tensor using one or more integers expresses a projection operation, which can be
assigned to or from a lower Rank tensor of the appropriate shape. Using a
projection in a compound Tensor exprssion does not automatically create a copy
of the underlying data, it merely restrict enumeration of the index space when
evaluating the expression. Some examples of binding operations can be seen
below.

```
Tensor<4,2,double> A{};

// Fully projected
A(1,2,0,1) = 1.0;        // assigns single element
auto d = A(1,2,0,1);     // reads single element

// Partial projection
constexpr const Index<'i'> i;                /// @see ttl/Index.h
constexpr const Index<'j'> j;
Tensor <2,2,double> P = A(i,2,1,j);          // copies data
Tensor <2,2,double> Q = P(i,j) + A(1,i,j,0); // restricts enumeration
```

Raw Tensors are not valid types for use in most expressions (there are some
Library operations that can operate on certain tensor types directly, like
transposes). Binding a Tensor provides access to the entire suite of
Expressions.

```
constexpr const Index<'i'> i;        // @see ttk/Index.h
constexpr const Index<'j'> j;
constexpr const Index<'k'> k;
constexpr const Index<'l'> l;

Tensor<2,2,double> A{}, B{};
Tensor<4,2,double> C;

C(i,j,k,l) = 1.9 * A(i,j) * B(k,l);  // outer product with scalar multiply
```

Tensor expressions with a tensor on the LHS force evaluation of the RHS. It can
be convenient to express complicated expressions as multipart terms. C++11 type
inference allows programmers to capture the intermediate expressions without
evaluating them, and they can then be combined (or reused) in other Expressions.

```
constexpr const Index<'i'> i;        // @see ttk/Index.h
constexpr const Index<'j'> j;
constexpr const Index<'k'> k;

Tensor<2,3,double> A{}, B{};
Tensor<2,3,double> C;

// The following two terms capture the matrix-matrix product expressions
// without evaluating anything. An optimizing compiler will produce no
// actual code for these two lines
auto AxB = A(i,j) * B(j,k);
auto BxA = B(i,j) * A(j,k);

// Use the terms in a compound expression. This works because the
// type identities of the 'i' and 'j' indices maintain their nature as
// part of the captured expressions.
C(i,k) = 0.5 * (AxB + BxA);

// For the CPU-based implementation, an optimizing compiler will produce a
// single triply-nested loop to evaluate this expression.
for (int i = 0; i < 3; ++i) {
  for (int k = 0; k < 3; k++) {
    C(i,k) = 0.0;
    for (int j = 0; i < 3; ++j) {
      C(i,k) += 0.5 * (A(i,j) * B(j,k) + B(i,j) * A(j,k));
    }
  }
}
```

If the scalar type of a Tensor is a pointer type, then the Tensor must be
initialized with a reference to an external buffer of the underlying type. In
this case, Tensor operations will effect that external storage. This form of
externally allocated Tensor storage can be useful when interacting with legacy
code, or when custom allocation is required (e.g., CUDA-allocated memory, or
pinned network memory).

```
double external[3][3][3][3];
Tensor<4,3,double*> A{external};
A(1,2,0,1) = 1.0;
assert(external[1][2][0][1] == 1.0);
```

Tensors with external storage can be used in any context in which a Tensor with
internal storage can be used.

## Expressions

TTL expressions are single C++ statements (potentially including captured
terms), with an assignment or increment operation. TTL supports a number of
basic expression types.

* Scalar Operations
```
s * A(i,j,k)
```
* Tensor Products
```
A(i,j) * B(k,l)
```
* Arbitrary Tensor Contractions
```
A(i,j,k,l) * B(i,k)
```
* Arbitrary Projections
```
A(i,2,k)
```
* Traces (self contraction)
```
A(i,i)
```
* Arbitrary Permutations
```
A(i,j,k).to(k,j,i) 
```
* Constant Expressions
These expressions are constant and do not allocate space when used in
expressions. They simply statically materialize a `1` or `0` based on the index
requested. A constant expression can be assigned to a compatibly shaped
Tensor. In normal usage as part of a compound expression, the manifold space
(Tensor dimension, `D`) can be inferred for these expressions.
  * A zero expression
```
zero<D>(i,j)
```
  * The delta expression 
```
identity<D>(i,j,k,l)
```
  * The epsilon expression
```
epsilon<D>(i,j)
```
  * The delta expression
```
delta<D>(i,j)
```

## Library Functions

In addition to basic expressions, TTL contains some small number of library
functions that can be applied to square matrices (rank 2 Tensors), or any Tensor
shape that can be reinterpreted as a square matrix.

* tranpose
* determinant
* inverse
* solve (`Ax=b`)

