A Tensor Template Library

*** Configuration

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


*** Mac OS X

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
