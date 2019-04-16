#pragma once

namespace ttl {
namespace mp {
template <bool B, class T, class>
struct iif {
  enum : bool { value = true };
  using type = T;
};

template <class T, class F>
struct iif<false, T, F> {
  enum : bool { value = false };
  using type = F;
};

template <bool B, class T, class F>
using iif_t = typename iif<B, T, F>::type;
} // namespace mp
} // namespace ttl
