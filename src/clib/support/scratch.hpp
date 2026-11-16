#ifndef SUPPORT_SCRATCH_HPP
#define SUPPORT_SCRATCH_HPP

#include <cstddef>
#include <vector>

// Zero-filled scratch vector that is reused between calls: it only allocates
// when it has to grow. thread_local keeps it safe under OpenMP. Not for
// functions that can re-enter themselves.
#define GRIMPL_SCRATCH_VECTOR(type, name, n)                                  \
  static thread_local std::vector<type> name;                                 \
  name.assign(static_cast<std::size_t>(n), type())

#endif  // SUPPORT_SCRATCH_HPP
