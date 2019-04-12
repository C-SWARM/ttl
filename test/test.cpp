#include <array>
#include <iostream>
#include <type_traits>

namespace ttl {
template <char Id>
class Index {
};

template <class T>
struct is_index {
  using type = std::true_type;
  enum : bool { value = false };
};

template <char Id>
struct is_index<Index<Id>> {
  using type = std::true_type;
  enum : bool { value = true };
};

template <class T>
struct is_scalar {
  enum : bool { value = std::is_arithmetic<T>::value };
};

template <class T>
struct is_integer {
  enum : bool { value = std::numeric_limits<T>::is_integer };
};

template <class T>
struct is_valid {
  enum : bool { value = is_index<T>::value or is_integer<T>::value };
};

template <template <class> class, class...>
struct all {
  enum : bool { value = true };
};

template <template <class> class P, class T, class... U>
struct all<P, T, U...> {
  enum : bool { value = P<T>::value and all<P, U...>::value };
};

template <class... U>
using all_scalar = all<is_scalar, U...>;

template <class... U>
using all_integer = all<is_integer, U...>;

template <class... U>
using all_valid = all<is_valid, U...>;

template <class To, class... U>
struct all_convertible {
  enum : bool { value = true };
};

template <class To, class From, class... U>
struct all_convertible<To, From, U...> {
  enum : bool { value = std::is_convertible<From, To>::value and
                        all_convertible<To, U...>::value };
};

template <class Expr, class Index>
class Bind {
 public:
  Bind(Expr expr, Index index) : expr_(expr), index_(std::move(index)) {
  }

  auto operator()(Index index) const {
    return expr_(index);
  }

 private:
  Expr expr_;
  Index index_;
};

template <class Expr, class Index>
Bind<Expr, Index> make_bind(Expr&& expr, Index index) {
  return { std::forward<Expr>(expr), std::move(index) };
}

template <class Lhs, class Rhs, class Op>
class BinaryOp {
 public:
  BinaryOp(Lhs lhs, Rhs rhs, Op op) : lhs_(std::move(lhs)),
                                      rhs_(std::move(rhs)),
                                      op_(std::move(op))
  {
  }

  template <class Index>
  auto operator()(Index index) const {
    return op_(lhs_(index), rhs_(index));
  }

 private:
  Lhs lhs_;
  Rhs rhs_;
  Op op_;
};

template <class Lhs, class Rhs, class Op>
BinaryOp<Lhs, Rhs, Op> make_binary_op(Lhs lhs, Rhs rhs, Op op) {
  return { std::move(lhs), std::move(rhs), std::move(op) };
}

template <class Expr, class Op>
class UnaryOp {
 public:
  UnaryOp(Expr expr, Op op) : expr_(std::move(expr)), op_(std::move(op)) {
  }

  template <class Index>
  auto operator()(Index index) const {
    return op_(expr_(index));
  }

 private:
  Expr expr_;
  Op op_;
};

template <class Expr, class Op>
UnaryOp<Expr, Op> make_unary_op(Expr expr, Op op) {
  return { std::move(expr), std::move(op) };
}

template <class Expr>
auto operator-(Expr expr) {
  return make_unary_op(std::move(expr), [](auto e) {
      return -e;
    });
}

template <class Lhs, class Rhs,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
auto operator/(Lhs lhs, Rhs rhs) {
  return make_unary_op(std::move(lhs), [r=std::move(rhs)](auto l) {
      return l / r;
    });
}

template <class Lhs, class Rhs,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
auto operator%(Lhs lhs, Rhs rhs) {
  return make_unary_op(std::move(lhs), [r=std::move(rhs)](auto l) {
      return l % r;
    });
}

template <class Lhs, class Rhs>
auto operator+(Lhs lhs, Rhs rhs) {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
      return l + r;
    });
}

template <class Lhs, class Rhs>
auto operator-(Lhs lhs, Rhs rhs) {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
      return l - r;
    });
}

template <class T, class Op, size_t... Is>
constexpr auto apply(T tuple, Op&& op, std::index_sequence<Is...>) {
  return op(std::get<Is>(tuple)...);
}

template <class... T, class Op>
constexpr auto apply(std::tuple<T...> tuple, Op&& op) {
  return apply(tuple, std::forward<Op>(op), std::make_index_sequence<sizeof...(T)>());
}

template <int Rank, int Dimension, class ScalarType>
class Tensor {
  static constexpr size_t pow(size_t r) {
    return (r) ? D * pow(r - 1) : 1;
  }

  static constexpr size_t size() {
    return pow(Rank);
  }

 public:
  static constexpr int R = Rank;
  static constexpr int D = Dimension;
  static constexpr int N = size();
  using T = ScalarType;

  static_assert(std::is_arithmetic<ScalarType>::value,
                "Tensors require fundamental scalar type");

  constexpr Tensor() noexcept = default;
  constexpr Tensor(const Tensor&) noexcept = default;
  constexpr Tensor(Tensor&&) noexcept = default;

  template <class... U,
            class = std::enable_if_t<all_convertible<T, U...>::value and N == sizeof...(U)>>
  constexpr Tensor(U... args) noexcept : data{static_cast<T>(args)...} {
  }

  constexpr Tensor& operator=(Tensor rhs) & noexcept {
    std::swap(data, rhs.data);
    return *this;
}

  template <class Rhs>
  constexpr Tensor& operator=(Rhs rhs) & noexcept {
    return *this;
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr const T& operator()(std::tuple<U...> index) const noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(apply(index, inRange));
    return data[apply(index, rowMajor)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(std::tuple<U...> index) noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(apply(index, inRange));
    return data[apply(index, rowMajor)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr const T& operator()(U... index) const& noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(inRange(index...));
    return data[rowMajor(index...)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(U... index) & noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(inRange(index...));
    return data[rowMajor(index...)];
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) const& noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return make_bind(*this, std::make_tuple(index...));
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) && noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return make_bind(std::move(*this), std::make_tuple(index...));
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) & noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return make_bind(*this, std::make_tuple(index...));
  }

 private:
  static constexpr bool inRange() {
    return true;
  }

  template <class T, class... U>
  static constexpr bool inRange(T car, U... cdr) {
    return 0 <= car and car < D and inRange(cdr...);
  }

  static constexpr size_t rowMajor() {
    return 0;
  }

  template <class T, class... U>
  static constexpr size_t rowMajor(T car, U... cdr) {
    return car * pow(sizeof...(U)) + rowMajor(cdr...);
  }

  std::array<T, size()> data = {};
};
}

int main() {
  constexpr ttl::Index<'i'> i;
  ttl::Tensor<2, 2, double> A;

  A(0,0) = 1;
  A(0,1) = 2;
  A(1,0) = 3;
  A(1,1) = 4;

  ttl::Tensor<2, 2, double> B = { 1, 2, 3, 4 };
  B = A;
  B = A(i,i);

  return 0;
}
