#pragma once

#include "ttl2/Algorithms.hpp"
#include "ttl2/Evaluate.hpp"
#include "ttl2/Operators.hpp"
#include "ttl2/mp.hpp"
#include "ttl/mp/non_integer.hpp"
#include "ttl/mp/unique.hpp"
#include "ttl/mp/duplicate.hpp"
#include <type_traits>
#include <utility>

namespace ttl {
template <class ChildIndexType, class ChildType>
class Bind {
 public:
  using    is_expression = void;
  using            Child = std::remove_reference_t<ChildType>;
  using       ChildIndex = ChildIndexType;
  using            Index = mp::non_integer_t<ChildIndexType>;
  using                T = typename Child::T;
  static constexpr int R = std::tuple_size<Index>::value;
  static constexpr int D = Child::D;

  constexpr Bind(ChildIndex index, ChildType op) noexcept
      : index_(index),
        op_(op)
  {
  }

  /// Evaluation of the child operation for a particular index.
  ///
  /// These just transform the index using the index that we captured and pass
  /// it along, for const Lvalue, Lvalue, and Rvalue contexts.
  /// @{
  constexpr const T& operator()(Index index) const & noexcept {
    return evaluate(index);
  }

  constexpr T& operator()(Index index) && noexcept {
    return evaluate(index);
  }

  constexpr T& operator()(Index index) & noexcept {
    return evaluate(index);
  }
  /// @}

  /// If the bind's rank is 0, then it can appear in a scalar context.
  /// @{
  constexpr operator iif_t<R == 0, const T&, void>() const & noexcept {
    return evaluate(Index{});
  }

  constexpr operator iif_t<R == 0, T&, void>() && noexcept {
    return evaluate(Index{});
  }

  constexpr operator iif_t<R == 0, T&, void>() & noexcept {
    return evaluate(Index{});
  }
  /// @}

  /// Enable access to bound tensors through integers.
  /// @{
  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr const T& operator()(U... index) const& noexcept {
    Index i = {std::move(index)...};
    return evaluate(i);
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(U... index) && noexcept {
    Index i = {std::move(index)...};
    return evaluate(i);
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(U... index) & noexcept {
    Index i = {std::move(index)...};
    return evaluate(i);
  }
  // @}

  /// This satisfies both move and copy assignment.
  constexpr Bind& operator=(Bind rhs) & noexcept {
    std::swap(index_, rhs.index_);
    std::swap(op_, rhs.op_);
    return *this;
  }

  /// Assignment from a compatible expression evaluates the right-hand-side.
  /// @{
  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator=(Rhs rhs) && noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) = r(index);
    });
    return *this;
  }

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator=(Rhs rhs) & noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) = r(index);
    });
    return *this;
  }
  /// @}

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator+=(Rhs rhs) && noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) += r(index);
    });
    return *this;
  }

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator+=(Rhs rhs) & noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) += r(index);
    });
    return *this;
  }

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator-=(Rhs rhs) && noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) -= r(index);
    });
    return *this;
  }

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator-=(Rhs rhs) & noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, Index>([this, r=std::move(rhs)](Index index) {
      evaluate(index) -= r(index);
    });
    return *this;
  }

  // template <class... U,
  //           class = std::enable_if_t<!all_integer<U...>::value>>
  // constexpr Bind<Index, Bind, std::tuple<U...>> to(U... index) const & noexcept {
  //   return { std::make_tuple(index...), *this };
  // }

  // template <class... U,
  //           class = std::enable_if_t<!all_integer<U...>::value>>
  // constexpr Bind<Index, Bind, std::tuple<U...>> to(U... index) && noexcept {
  //   return { std::make_tuple(index...), *this };
  // }

  // template <class... U,
  //           class = std::enable_if_t<!all_integer<U...>::value>>
  // constexpr Bind<Index, Bind, std::tuple<U...>> to(U... index) & noexcept {
  //   return { std::make_tuple(index...), *this };
  // }

 private:
  /// Evaluate a fully specialized index.
  ///
  /// These transform the external Index type to the ChildIndex type, which
  /// handle any dimensions that have been explicitly sliced.
  ///
  /// @{
  constexpr const T& evaluate(Index index) const & noexcept {
    return op_(select(index_, index));
  }

  constexpr T& evaluate(Index index) && noexcept {
    return op_(select(index_, index));
  }

  constexpr T& evaluate(Index index) & noexcept {
    return op_(select(index_, index));
  }
  /// @}

  ChildIndex index_;
  ChildType op_;
};

/// The primary constructor for a basic bind operation.
///
/// This determines if the Index tuple indicates that we need self contraction
/// by looking for duplicate non-integer types in the index, uses tag dispatch
/// to inject a trace node if necessary.
///
/// @todo[c++17] This would be much simpler with `if constexpr` as we could
///              return a Bind or a Trace directly without the intermediate
///              tag dispatch operation.
template <class Index, class Op>
constexpr auto bind(Index index, Op&& op) noexcept {
  using NonIntegerIndex = mp::non_integer_t<Index>;
  using           Inner = mp::duplicate_t<NonIntegerIndex>;
  constexpr  bool trace = std::tuple_size<Inner>::value != 0;
  using           Trace = std::integral_constant<bool, trace>;
  return bind(index, std::forward<Op>(op), Trace{});
}

/// Bind a tensor in the absence of any potential slicing.
///
/// This version of the bind() function is used when the client doesn't have a
/// physical index to bind to yet, but merely knows the type of the index to
/// accept. This occurs when the user assigns an expression to a tensor.
template <class Index, class Op>
constexpr auto bind(Op&& op) noexcept {
  return bind(Index{}, std::forward<Op>(op));
}

/// Create a basic bind node.
///
/// This bind expects to be called with tuples of `non_integer<Index>` type, and
/// will merge them with any slicing operations.
template <class Index, class Op>
constexpr Bind<Index, Op> bind(Index index, Op&& op, std::false_type) noexcept {
  return { index, std::forward<Op>(op) };
}

/// Create a trace node.
///
/// This overload creates a bind node and then puts a trace node on top of
/// it. The trace node performs a self contraction on the repeated indices
/// present in the Index.
template <class Index, class Op>
constexpr auto bind(Index index, Op&& op, std::true_type) noexcept {
  // Create the child bind operation.
  auto  child = bind(index, std::forward<Op>(op), std::false_type{});

  // Figure out the types we need to interact with for the trace. It exports an
  // "Outer" type, which is a tuple that it expects to be called with, and has
  // an "Inner" type which are the indices that need contraction, and then it
  // maps the resulting Union indices to the ChildIndex type.
  using Child      = decltype(child);
  using ChildIndex = typename Child::Index;
  using Outer      = mp::unique_t<ChildIndex>;
  using Inner      = mp::duplicate_t<ChildIndex>;
  using Union      = mp::cat_t<Outer, Inner>;

  // Take an outer index, append and contract along an inner index (creating a
  // sequence of union indices), and then transform them to call the bind.
  constexpr int D = Child::D;
  return make_expression<D, Outer>([op=std::move(child)](Outer outer) {
    return contract<D, Inner>(outer, [op=std::move(op)](Union index) {
      return op(select<ChildIndex>(index));
    });
  });
}
}
