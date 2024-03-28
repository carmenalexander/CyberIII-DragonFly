// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include <chrono>

#include "server/db_slice.h"
#include "server/io_utils.h"
#include "server/journal/journal.h"
#include "server/journal/serializer.h"
#include "server/rdb_save.h"

namespace dfly {

// Buffered single-shard journal streamer that listens for journal changes with a
// journal listener and writes them to a destination sink in a separate fiber.
class JournalStreamer : protected BufferedStreamerBase {
 public:
  JournalStreamer(journal::Journal* journal, Context* cntx)
      : BufferedStreamerBase{cntx->GetCancellation()},
        cntx_{cntx},
        journal_{journal},
        periodic_ping_(this) {
  }

  // Self referential.
  JournalStreamer(const JournalStreamer& other) = delete;
  JournalStreamer(JournalStreamer&& other) = delete;

  // Register journal listener and start writer in fiber.
  virtual void Start(io::Sink* dest, bool with_pings = false);

  // Must be called on context cancellation for unblocking
  // and manual cleanup.
  virtual void Cancel();

  using BufferedStreamerBase::GetTotalBufferCapacities;

 private:
  // Writer fiber that steals buffer contents and writes them to dest.
  void WriterFb(io::Sink* dest);
  virtual bool ShouldWrite(const journal::JournalItem& item) const {
    return true;
  }

  class PeriodicPing {
   public:
    explicit PeriodicPing(JournalStreamer* streamer) : streamer_(streamer) {
    }
    void MaybePing(bool allow_await);
    void Start();

    static constexpr std::chrono::seconds kLimit{2};
    friend JournalStreamer;

   private:
    std::chrono::system_clock::time_point start_time_;
    JournalStreamer* streamer_;
  };

  Context* cntx_;

  uint32_t journal_cb_id_{0};
  journal::Journal* journal_;
  uint64_t total_records_{0};

  util::fb2::Fiber write_fb_{};
  PeriodicPing periodic_ping_;
};

// Serializes existing DB as RESTORE commands, and sends updates as regular commands.
// Only handles relevant slots, while ignoring all others.
class RestoreStreamer : public JournalStreamer {
 public:
  RestoreStreamer(DbSlice* slice, SlotSet slots, journal::Journal* journal, Context* cntx);
  ~RestoreStreamer() override;

  void Start(io::Sink* dest, bool with_pings = false) override;
  // Cancel() must be called if Start() is called
  void Cancel() override;

  void SendFinalize();

  bool IsSnapshotFinished() const {
    return snapshot_finished_;
  }

 private:
  void OnDbChange(DbIndex db_index, const DbSlice::ChangeReq& req);
  bool ShouldWrite(const journal::JournalItem& item) const override;
  bool ShouldWrite(std::string_view key) const;
  bool ShouldWrite(SlotId slot_id) const;

  // Returns whether anything was written
  bool WriteBucket(PrimeTable::bucket_iterator it);
  void WriteEntry(string_view key, const PrimeValue& pv, uint64_t expire_ms);
  void WriteCommand(journal::Entry::Payload cmd_payload);

  DbSlice* db_slice_;
  uint64_t snapshot_version_ = 0;
  SlotSet my_slots_;
  Cancellation fiber_cancellation_;
  bool snapshot_finished_ = false;
};

}  // namespace dfly
