#pragma once

#include "ttl2/Evaluate.hpp"
#include "ttl2/mp.hpp"
#include "ttl/mp/non_integer.hpp"
#include <utility>

namespace ttl {
template <class IndexType, class ChildType>
class Bind {
 public:
  using    is_expression = void;
  using            Child = std::remove_reference_t<ChildType>;
  using            Index = mp::non_integer_t<IndexType>;
  using                T = typename Child::T;
  static constexpr int R = std::tuple_size<Index>::value;
  static constexpr int D = Child::D;

  constexpr Bind(IndexType index, ChildType op) noexcept
      : index_(index),
        op_(op)
  {
  }

  /// Evaluation of the child operation for a particular index.
  ///
  /// These just transform the index using the index that we captured and pass
  /// it along, for const L-value, L-value, and R-value contexts.
  /// @{
  constexpr auto& operator()(Index index) const & noexcept {
    return op_(index);
  }

  constexpr auto& operator()(Index index) && noexcept {
    return op_(index);
  }

  constexpr auto& operator()(Index index) & noexcept {
    return op_(index);
  }
  /// @}

  /// This satisfies both move and copy assignment.
  constexpr Bind& operator=(Bind rhs) & noexcept {
    std::swap(index_, rhs.index_);
    std::swap(op_, rhs.op_);
    return *this;
  }

  /// Assignment from a compatible expression evaluates the right-hand-side.
  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator=(Rhs rhs) && noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, 0, R>::op(Index{}, [this, r=std::move(rhs)](Index index) {
      op_(index) = r(index);
    });
    return *this;
  }

  /// Assignment from a compatible expression evaluates the right-hand-side.
  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Bind& operator=(Rhs rhs) & noexcept {
    constexpr auto D = dimension<Bind, Rhs>::value;
    constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
    static_assert(compatible, "Index must match for tensor assignment");
    forall<D, 0, R>::op(Index{}, [this, r=std::move(rhs)](Index index) {
      op_(index) = r(index);
    });
    return *this;
  }

 private:
  IndexType index_;
  ChildType op_;
};

template <class Index, class Op>
constexpr Bind<Index, Op> bind(Index index, Op&& op) noexcept {
  return { index, std::forward<Op>(op) };
}

template <class Index, class Op>
constexpr Bind<Index, Op> bind(Op&& op) noexcept {
  return { Index{}, std::forward<Op>(op) };
}
}
