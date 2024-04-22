// Copyright 2024, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/tx_base.h"

namespace dfly {

inline size_t ShardArgs::Size() const {
  size_t sz = 0;
  for (const auto& s : slices)
    sz += (s.second - s.first);
  return sz;
}

}  // namespace dfly
