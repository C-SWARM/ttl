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
$ cmake ../ttl -DCMAKE_INSTALL_PREFIX=$HOME/ttl-install
$ make -j
$ make test
$ make install
$ export C_INCLUDE_PATH="$HOME/ttl-install/include:$C_INCLUDE_PATH"
$ export CPLUS_INCLUDE_PATH="$HOME/ttl-install/include:$CPLUS_INCLUDE_PATH"
```
