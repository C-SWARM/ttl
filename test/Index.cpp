#include <ttl/ttl.h>

// Declare i, j, k index types.
using i = ttl::Index<'i'>;
using j = ttl::Index<'j'>;
using k = ttl::Index<'k'>;

// Build some base index lists based on i,j,k
using l_empty = std::tuple<         >;
using     l_i = std::tuple< i       >;
using     l_j = std::tuple< j       >;
using     l_k = std::tuple< k       >;
using    l_ij = std::tuple< i, j    >;
using    l_ji = std::tuple< j, i    >;
using    l_ik = std::tuple< i, k    >;
using    l_jk = std::tuple< j, k    >;
using   l_ijk = std::tuple< i, j, k >;
using   l_kij = std::tuple< k, i, j >;

// Test equivalentalency
static_assert(  std::is_same< l_i,     l_i >::value, "failed");
static_assert( !std::is_same< l_i,     l_j >::value, "failed");
static_assert(  std::is_same< l_ij,   l_ij >::value, "failed");
static_assert( !std::is_same< l_ij,   l_ji >::value, "failed");
static_assert(  std::is_same< l_ijk, l_ijk >::value, "failed");
static_assert( !std::is_same< l_ijk, l_kij >::value, "failed");

static_assert(  ttl::util::is_equivalent< l_i,     l_i >::value, "failed");
static_assert( !ttl::util::is_equivalent< l_i,     l_j >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ij,   l_ij >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ij,   l_ji >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ijk, l_ijk >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ijk, l_kij >::value, "failed");

// Create some joined lists
using  j_ij = ttl::expressions::union_< l_i,  l_j  >;
using  j_ji = ttl::expressions::union_< l_j,  l_i  >;
using j_ijk = ttl::expressions::union_< j_ij, l_k  >;
using j_kij = ttl::expressions::union_< l_k,  j_ij >;

// Test equivalentalency
static_assert(  std::is_same< l_ij,  j_ij  >::value, "failed");
static_assert( !std::is_same< l_ji,  j_ij  >::value, "failed");
static_assert(  std::is_same< l_ijk, j_ijk >::value, "failed");
static_assert( !std::is_same< l_ijk, j_kij >::value, "failed");

static_assert(  ttl::util::is_equivalent< l_ij,  j_ij  >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ji,  j_ij  >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ijk, j_ijk >::value, "failed");
static_assert(  ttl::util::is_equivalent< l_ijk, j_kij >::value, "failed");

// Create some intersections
using i_empty1 = ttl::expressions::intersection< l_i,   l_j   >;
using     i_i1 = ttl::expressions::intersection< l_i,   l_i   >;
using     i_i2 = ttl::expressions::intersection< l_i,   j_ij  >;
using     i_i3 = ttl::expressions::intersection< l_ji,  l_ik  >;
using    i_ij1 = ttl::expressions::intersection< j_ij,  l_ij  >;
using    i_ij2 = ttl::expressions::intersection< l_ij,  j_ji  >;
using    i_ij3 = ttl::expressions::intersection< l_ij,  j_ijk >;
using    i_ij4 = ttl::expressions::intersection< l_ijk, j_ji  >;

// Check intersections
static_assert( std::is_same< i_empty1, l_empty >::value, "failed");

static_assert( std::is_same< i_i1,   l_i >::value, "failed");
static_assert( std::is_same< i_i2,   l_i >::value, "failed");
static_assert( std::is_same< i_ij1, l_ij >::value, "failed");
static_assert( std::is_same< i_ij2, l_ij >::value, "failed");
static_assert( std::is_same< i_ij3, l_ij >::value, "failed");
static_assert( std::is_same< i_ij4, l_ij >::value, "failed");

// Create some symmetric differences
using x_empty1 = ttl::expressions::outer_type< l_i,   l_i   >;
using x_empty2 = ttl::expressions::outer_type< l_ij,  l_ji  >;
using x_empty3 = ttl::expressions::outer_type< l_ijk, j_kij >;
using    x_ij1 = ttl::expressions::outer_type< l_i,   l_j   >;
using    x_jk1 = ttl::expressions::outer_type< l_ji,  l_ik  >;
using    x_jk2 = ttl::expressions::outer_type< l_ijk, l_i   >;

// Check symmetric differences
static_assert( std::is_same< x_empty1, l_empty >::value, "failed");
static_assert( std::is_same< x_empty2, l_empty >::value, "failed");
static_assert( std::is_same< x_empty3, l_empty >::value, "failed");

static_assert( std::is_same< x_ij1, l_ij >::value, "failed");
static_assert( std::is_same< x_jk1, l_jk >::value, "failed");
static_assert( std::is_same< x_jk2, l_jk >::value, "failed");

int main() {
  return 0;
}
