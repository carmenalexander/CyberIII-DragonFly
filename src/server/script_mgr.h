// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include <absl/container/flat_hash_map.h>

#include <array>
#include <boost/fiber/mutex.hpp>

#include "server/conn_context.h"

namespace dfly {

class EngineShardSet;

// This class has a state through the lifetime of a server because it manipulates scripts
class ScriptMgr {
 public:
  ScriptMgr();

  void Run(CmdArgList args, ConnectionContext* cntx);

  // Insert script. Returns true if inserted new script.
  bool Insert(std::string_view sha, std::string_view body);

  // Get script body by sha, returns nullptr if not found.
  const char* Find(std::string_view sha) const;

  // Returns a list of all scripts in the database with their sha and body.
  std::vector<std::pair<std::string, std::string>> GetAll() const;

 private:
  void ExistsCmd(CmdArgList args, ConnectionContext* cntx) const;
  void LoadCmd(CmdArgList args, ConnectionContext* cntx);
  void ListCmd(ConnectionContext* cntx) const;
  void LatencyCmd(ConnectionContext* cntx) const;

 private:
  using ScriptKey = std::array<char, 40>;
  absl::flat_hash_map<ScriptKey, std::unique_ptr<char[]>> db_;  // protected by mu_
  mutable ::boost::fibers::mutex mu_;
};

}  // namespace dfly
