#include <ttl/ttl.h>

// Declare i, j, k indices
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

// Build some base index lists based on i,j,k
static auto l_empty = ttl::index_list_pack();
static auto l_i = ttl::index_list_pack(i);
static auto l_j = ttl::index_list_pack(j);
static auto l_k = ttl::index_list_pack(k);
static auto l_ij = ttl::index_list_pack(i, j);
static auto l_ji = ttl::index_list_pack(j, i);
static auto l_ik = ttl::index_list_pack(i, k);
static auto l_jk = ttl::index_list_pack(j, k);
static auto l_ijk = ttl::index_list_pack(i, j, k);
static auto l_kij = ttl::index_list_pack(k, i, j);

// Test equivalency
static_assert(ttl::index_list_eq(l_i, l_i), "failed");
static_assert(ttl::index_list_equiv(l_i, l_i), "failed");
static_assert(!ttl::index_list_eq(l_i, l_j), "failed");
static_assert(!ttl::index_list_equiv(l_i, l_j), "failed");
static_assert(ttl::index_list_eq(l_ij, l_ij), "failed");
static_assert(ttl::index_list_equiv(l_ij, l_ij), "failed");
static_assert(!ttl::index_list_eq(l_ij, l_ji), "failed");
static_assert(ttl::index_list_equiv(l_ij, l_ji), "failed");
static_assert(ttl::index_list_eq(l_ijk, l_ijk), "failed");
static_assert(ttl::index_list_equiv(l_ijk, l_ijk), "failed");
static_assert(!ttl::index_list_eq(l_ijk, l_kij), "failed");
static_assert(ttl::index_list_equiv(l_ijk, l_kij), "failed");

// Create some joined lists
static auto j_ij = ttl::index_list_or(l_i,l_j);
static auto j_ji = ttl::index_list_or(l_j,l_i);
static auto j_ijk = ttl::index_list_or(j_ij,l_k);
static auto j_kij = ttl::index_list_or(l_k,j_ij);

// Test equivalency
static_assert(ttl::index_list_eq(l_ij, j_ij), "failed");
static_assert(ttl::index_list_equiv(l_ij, j_ij), "failed");
static_assert(!ttl::index_list_eq(l_ji, j_ij), "failed");
static_assert(ttl::index_list_equiv(l_ji, j_ij), "failed");
static_assert(ttl::index_list_eq(l_ijk, j_ijk), "failed");
static_assert(ttl::index_list_equiv(l_ijk, j_ijk), "failed");
static_assert(!ttl::index_list_eq(l_ijk, j_kij), "failed");
static_assert(ttl::index_list_equiv(l_ijk, j_kij), "failed");

// Create some intersections
static auto i_empty1 =  ttl::index_list_and(l_i,l_j);
static auto i_i1 =  ttl::index_list_and(l_i,l_i);
static auto i_i2 =  ttl::index_list_and(l_i,j_ij);
static auto i_i3 =  ttl::index_list_and(l_ji,l_ik);
static auto i_ij1 = ttl::index_list_and(j_ij,l_ij);
static auto i_ij2 = ttl::index_list_and(l_ij,j_ji);
static auto i_ij3 = ttl::index_list_and(l_ij,j_ijk);
static auto i_ij4 = ttl::index_list_and(l_ijk,j_ji);

// Check intersections
static_assert(ttl::index_list_eq(i_empty1, l_empty), "failed");
static_assert(ttl::index_list_eq(i_i1, l_i), "failed");
static_assert(ttl::index_list_eq(i_i2, l_i), "failed");
static_assert(ttl::index_list_eq(i_ij1, l_ij), "failed");
static_assert(ttl::index_list_eq(i_ij2, l_ij), "failed");
static_assert(ttl::index_list_eq(i_ij3, l_ij), "failed");
static_assert(ttl::index_list_eq(i_ij4, l_ij), "failed");

// Create some symmetric differences
static auto x_empty1 = ttl::index_list_xor(l_i, l_i);
static auto x_empty2 = ttl::index_list_xor(l_ij, l_ji);
static auto x_empty3 = ttl::index_list_xor(l_ijk, j_kij);
static auto x_ij1 = ttl::index_list_xor(l_i, l_j);
static auto x_jk1 = ttl::index_list_xor(l_ji, l_ik);
static auto x_jk2 = ttl::index_list_xor(l_ijk, l_i);

// Check symmetric differences
static_assert(ttl::index_list_eq(x_empty1, l_empty), "failed");
static_assert(ttl::index_list_eq(x_empty2, l_empty), "failed");
static_assert(ttl::index_list_eq(x_empty3, l_empty), "failed");
static_assert(ttl::index_list_eq(x_ij1, l_ij), "failed");
static_assert(ttl::index_list_eq(x_jk1, l_jk), "failed");
static_assert(ttl::index_list_eq(x_jk2, l_jk), "failed");

int main() {
  return 0;
}
