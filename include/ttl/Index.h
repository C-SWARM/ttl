// -*- C++ -*-
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

namespace ttl {

/// This is an index template that can be used to index a tensor.
///
/// Each class template is parameterized by a character. Index values that have
/// the same class are assumed to be the same index.
///
/// @code
///   Index<'a'> i;
///   Index<'a'> j;
///   Tensor<2,int,2> M,N;
///   M(i,j) == M(i,i);
/// @code
///
/// Internally, ttl uses indexes as loop control variables to iterate through
/// contractions and assignment operations.
template <char ID>
struct Index {
  static constexpr char id = ID;

  constexpr Index() : value_(0) {
  }

  constexpr explicit Index(int value) : value_(value) {
  }

  Index(const Index&) = default;
  Index(Index&&) = default;
  Index& operator=(const Index&) = default;
  Index& operator=(Index&&) = default;

  Index& operator=(int i) {
    value_ = i;
    return *this;
  }

  Index& set(int i) {
    value_ = i;
    return *this;
  }

  constexpr operator int() const {
    return value_;
  }

  constexpr bool operator<(int e) const {
    return value_ < e;
  }

  Index& operator++() {
    ++value_;
    return *this;
  }

 private:
  int value_;
};
} // namespace ttl

#endif // TTL_INDEX_H
