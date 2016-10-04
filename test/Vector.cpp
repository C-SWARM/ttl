#include <ttl/ttl.h>

using std::is_same;
template <char ID> using Index = ttl::Index<ID>;
using Vector = ttl::Tensor<1, double, 3>;

int main() {
  static constexpr Index<'i'> i;
  static constexpr Index<'j'> j;

  Vector v;
  {
    auto t = v(i);
    auto u = v(j);

    static_assert(!is_same<decltype(v(i)), decltype(v(j))>::value,
                  "types should be different");
    static_assert(is_same<decltype(v(i)), decltype(v(i))>::value,
                  "types should match");
  }

  // {
  //   auto v = v(i) + v(i);
  // }

  return 0;
}
