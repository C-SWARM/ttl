#include <ttl/ttl.h>

// Declare i, j, k index types.
using i = ttl::Index<'i'>;
using j = ttl::Index<'j'>;
using k = ttl::Index<'k'>;

// Build some base index lists based on i,j,k
using l_empty = ttl::Pack<         >;
using     l_i = ttl::Pack< i       >;
using     l_j = ttl::Pack< j       >;
using     l_k = ttl::Pack< k       >;
using    l_ij = ttl::Pack< i, j    >;
using    l_ji = ttl::Pack< j, i    >;
using    l_ik = ttl::Pack< i, k    >;
using    l_jk = ttl::Pack< j, k    >;
using   l_ijk = ttl::Pack< i, j, k >;
using   l_kij = ttl::Pack< k, i, j >;

// Test equivalentalency
static_assert(  ttl::is_equal< l_i,     l_i >::value, "failed");
static_assert( !ttl::is_equal< l_i,     l_j >::value, "failed");
static_assert(  ttl::is_equal< l_ij,   l_ij >::value, "failed");
static_assert( !ttl::is_equal< l_ij,   l_ji >::value, "failed");
static_assert(  ttl::is_equal< l_ijk, l_ijk >::value, "failed");
static_assert( !ttl::is_equal< l_ijk, l_kij >::value, "failed");

static_assert(  ttl::is_equivalent< l_i,     l_i >::value, "failed");
static_assert( !ttl::is_equivalent< l_i,     l_j >::value, "failed");
static_assert(  ttl::is_equivalent< l_ij,   l_ij >::value, "failed");
static_assert(  ttl::is_equivalent< l_ij,   l_ji >::value, "failed");
static_assert(  ttl::is_equivalent< l_ijk, l_ijk >::value, "failed");
static_assert(  ttl::is_equivalent< l_ijk, l_kij >::value, "failed");

// Create some joined lists
using  j_ij = ttl::unite< l_i,  l_j  >;
using  j_ji = ttl::unite< l_j,  l_i  >;
using j_ijk = ttl::unite< j_ij, l_k  >;
using j_kij = ttl::unite< l_k,  j_ij >;

// Test equivalentalency
static_assert(  ttl::is_equal< l_ij,  j_ij  >::value, "failed");
static_assert( !ttl::is_equal< l_ji,  j_ij  >::value, "failed");
static_assert(  ttl::is_equal< l_ijk, j_ijk >::value, "failed");
static_assert( !ttl::is_equal< l_ijk, j_kij >::value, "failed");

static_assert(  ttl::is_equivalent< l_ij,  j_ij  >::value, "failed");
static_assert(  ttl::is_equivalent< l_ji,  j_ij  >::value, "failed");
static_assert(  ttl::is_equivalent< l_ijk, j_ijk >::value, "failed");
static_assert(  ttl::is_equivalent< l_ijk, j_kij >::value, "failed");

// Create some intersections
using i_empty1 = ttl::intersect< l_i,   l_j   >;
using     i_i1 = ttl::intersect< l_i,   l_i   >;
using     i_i2 = ttl::intersect< l_i,   j_ij  >;
using     i_i3 = ttl::intersect< l_ji,  l_ik  >;
using    i_ij1 = ttl::intersect< j_ij,  l_ij  >;
using    i_ij2 = ttl::intersect< l_ij,  j_ji  >;
using    i_ij3 = ttl::intersect< l_ij,  j_ijk >;
using    i_ij4 = ttl::intersect< l_ijk, j_ji  >;

// Check intersections
static_assert( ttl::is_equal< i_empty1, l_empty >::value, "failed");

static_assert( ttl::is_equal< i_i1,   l_i >::value, "failed");
static_assert( ttl::is_equal< i_i2,   l_i >::value, "failed");
static_assert( ttl::is_equal< i_ij1, l_ij >::value, "failed");
static_assert( ttl::is_equal< i_ij2, l_ij >::value, "failed");
static_assert( ttl::is_equal< i_ij3, l_ij >::value, "failed");
static_assert( ttl::is_equal< i_ij4, l_ij >::value, "failed");

// Create some symmetric differences
using x_empty1 = ttl::symdif< l_i,   l_i   >;
using x_empty2 = ttl::symdif< l_ij,  l_ji  >;
using x_empty3 = ttl::symdif< l_ijk, j_kij >;
using    x_ij1 = ttl::symdif< l_i,   l_j   >;
using    x_jk1 = ttl::symdif< l_ji,  l_ik  >;
using    x_jk2 = ttl::symdif< l_ijk, l_i   >;

// Check symmetric differences
static_assert( ttl::is_equal< x_empty1, l_empty >::value, "failed");
static_assert( ttl::is_equal< x_empty2, l_empty >::value, "failed");
static_assert( ttl::is_equal< x_empty3, l_empty >::value, "failed");

static_assert( ttl::is_equal< x_ij1, l_ij >::value, "failed");
static_assert( ttl::is_equal< x_jk1, l_jk >::value, "failed");
static_assert( ttl::is_equal< x_jk2, l_jk >::value, "failed");

int main() {
  return 0;
}
