// Copyright 2023, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//
#pragma once

#include "server/protocol_client.h"

namespace dfly {
class ClusterShardMigration;

class Service;
class ClusterFamily;

// The main entity on the target side that manage slots migration process
// Creates initial connection between the target and source node,
// manage migration process state and data
class ClusterSlotMigration : private ProtocolClient {
 public:
  struct Info {
    std::string host;
    uint16_t port;
  };

  ClusterSlotMigration(ClusterFamily* cl_fm, std::string host_ip, uint16_t port, Service* se,
                       SlotRanges slots);
  ~ClusterSlotMigration();

  // Initiate connection with source node and create migration fiber
  // will be refactored in the future
  std::error_code Init(uint32_t sync_id, uint32_t shards_num);

  void StartFlow(uint32_t shard, io::Source* source);

  Info GetInfo() const;
  uint32_t GetSyncId() const {
    return sync_id_;
  }

  // Remote sync ids uniquely identifies a sync *remotely*. However, multiple remote sources can
  // use the same id, so we need a local id as well.
  uint32_t GetLocalSyncId() const {
    return local_sync_id_;
  }

  MigrationState GetState() const {
    return state_;
  }

  void Stop();

  const SlotRanges& GetSlots() const {
    return slots_;
  }

 private:
  void MainMigrationFb();
  // Creates flows, one per shard on the source node and manage migration process
  std::error_code InitiateSlotsMigration();

  // may be called after we finish all flows
  bool IsFinalized() const;

 private:
  ClusterFamily* cluster_family_;
  Service& service_;
  util::fb2::Mutex flows_op_mu_;
  std::vector<std::unique_ptr<ClusterShardMigration>> shard_flows_;
  SlotRanges slots_;
  uint32_t source_shards_num_ = 0;
  uint32_t sync_id_ = 0;
  uint32_t local_sync_id_ = 0;
  MigrationState state_ = MigrationState::C_NO_STATE;
  std::vector<std::vector<unsigned>> partitions_;

  util::fb2::Fiber sync_fb_;
};

}  // namespace dfly
