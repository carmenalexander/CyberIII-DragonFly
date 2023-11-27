// Copyright 2023, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//
#pragma once

#include "server/protocol_client.h"

namespace dfly {

class ClusterSlotMigration : ProtocolClient {
 public:
  ClusterSlotMigration(std::string source_host, uint16_t port,
                       std::vector<ClusterConfig::SlotRange> slots);
  ~ClusterSlotMigration();

  std::error_code Start(ConnectionContext* cntx);

 private:
  std::error_code Greet();
  std::vector<ClusterConfig::SlotRange> slots_;
  size_t flows_num_ = 0;
};

}  // namespace dfly
