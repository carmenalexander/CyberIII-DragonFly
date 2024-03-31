// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/journal/journal_slice.h"

#include <absl/container/inlined_vector.h>
#include <absl/flags/flag.h>
#include <absl/strings/escaping.h>
#include <absl/strings/str_cat.h>
#include <fcntl.h>

#include <filesystem>

#include "base/function2.hpp"
#include "base/logging.h"
#include "server/journal/serializer.h"

ABSL_FLAG(uint32_t, shard_repl_backlog_len, 1 << 10,
          "The length of the circular replication log per shard");

namespace dfly {
namespace journal {
using namespace std;
using namespace util;
namespace fs = std::filesystem;

namespace {

/*
string ShardName(std::string_view base, unsigned index) {
  return absl::StrCat(base, "-", absl::Dec(index, absl::kZeroPad4), ".log");
}

uint32_t NextPowerOf2(uint32_t x) {
  if (x < 2) {
    return 1;
  }
  int log = 32 - __builtin_clz(x - 1);
  return 1 << log;
}

*/

}  // namespace

#define CHECK_EC(x)                                                                 \
  do {                                                                              \
    auto __ec$ = (x);                                                               \
    CHECK(!__ec$) << "Error: " << __ec$ << " " << __ec$.message() << " for " << #x; \
  } while (false)

JournalSlice::JournalSlice() {
}

JournalSlice::~JournalSlice() {
  // CHECK(!shard_file_);
}

void JournalSlice::Init(unsigned index) {
  if (ring_buffer_)  // calling this function multiple times is allowed and it's a no-op.
    return;

  slice_index_ = index;
  ring_buffer_.emplace(2);
}

#if 0
std::error_code JournalSlice::Open(std::string_view dir) {
  CHECK(!shard_file_);
  DCHECK_NE(slice_index_, UINT32_MAX);

  fs::path dir_path;

  if (dir.empty()) {
  } else {
    dir_path = dir;
    error_code ec;

    fs::file_status dir_status = fs::status(dir_path, ec);
    if (ec) {
      if (ec == errc::no_such_file_or_directory) {
        fs::create_directory(dir_path, ec);
        dir_status = fs::status(dir_path, ec);
      }
      if (ec)
        return ec;
    }
    // LOG(INFO) << int(dir_status.type());
  }

  dir_path.append(ShardName("journal", slice_index_));
  shard_path_ = dir_path;

  // For file integrity guidelines see:
  // https://lwn.net/Articles/457667/
  // https://www.evanjones.ca/durability-filesystem.html
  // NOTE: O_DSYNC is omitted.
  constexpr auto kJournalFlags = O_CLOEXEC | O_CREAT | O_TRUNC | O_RDWR;
  io::Result<unique_ptr<LinuxFile>> res = OpenLinux(shard_path_, kJournalFlags, 0666);
  if (!res) {
    return res.error();
  }
  DVLOG(1) << "Opened journal " << shard_path_;

  shard_file_ = std::move(res).value();
  file_offset_ = 0;
  status_ec_.clear();

  return error_code{};
}

error_code JournalSlice::Close() {
  VLOG(1) << "JournalSlice::Close";

  CHECK(shard_file_);
  lameduck_ = true;

  auto ec = shard_file_->Close();

  DVLOG(1) << "Closing " << shard_path_;
  LOG_IF(ERROR, ec) << "Error closing journal file " << ec;
  shard_file_.reset();

  return ec;
}
#endif

bool JournalSlice::IsLSNInBuffer(LSN lsn) const {
  DCHECK(ring_buffer_);

  if (ring_buffer_->empty()) {
    return false;
  }
  return (*ring_buffer_)[0].lsn <= lsn && lsn <= ((*ring_buffer_)[ring_buffer_->size() - 1].lsn);
}

std::string_view JournalSlice::GetEntry(LSN lsn) const {
  DCHECK(ring_buffer_ && IsLSNInBuffer(lsn));
  auto start = (*ring_buffer_)[0].lsn;
  DCHECK((*ring_buffer_)[lsn - start].lsn == lsn);
  return (*ring_buffer_)[lsn - start].data;
}

void JournalSlice::AddLogRecord(const Entry& entry, bool await) {
  optional<FiberAtomicGuard> guard;
  if (!await) {
    guard.emplace();  // Guard is non-movable/copyable, so we must use emplace()
  }

  DCHECK(ring_buffer_);

  JournalItem dummy;
  JournalItem* item;
  if (entry.opcode == Op::NOOP) {
    item = &dummy;
    item->lsn = -1;
    item->opcode = entry.opcode;
    item->data = "";
    item->slot = entry.slot;
  } else {
    FiberAtomicGuard fg;
    // GetTail gives a pointer to a new tail entry in the buffer, possibly overriding the last entry
    // if the buffer is full.
    item = &dummy;
    item->opcode = entry.opcode;
    item->lsn = lsn_++;
    item->slot = entry.slot;

    io::BufSink buf_sink{&ring_serialize_buf_};
    JournalWriter writer{&buf_sink};
    writer.Write(entry);

    item->data = io::View(ring_serialize_buf_.InputBuffer());
    ring_serialize_buf_.Clear();
    VLOG(2) << "Writing item [" << item->lsn << "]: " << entry.ToString();
  }

#if 0
    if (shard_file_) {
      string line = absl::StrCat(item.lsn, " ", entry.txid, " ", entry.opcode, "\n");
      error_code ec = shard_file_->Write(io::Buffer(line), file_offset_, 0);
      CHECK_EC(ec);
      file_offset_ += line.size();
    }
#endif
  const auto now = time(nullptr);
  std::optional<JournalItem> ping_item;
  if ((now - start_time_) > 2) {
    journal::Entry entry(0, journal::Op::PING, 0, 0, nullopt, {});
    io::BufSink buf_sink{&ring_serialize_buf_};
    JournalWriter writer{&buf_sink};
    writer.Write(entry);
    writer.Write(lsn_++);
    item->data = io::View(ring_serialize_buf_.InputBuffer());
    ring_serialize_buf_.Clear();
    start_time_ = time(nullptr);
  }

  // TODO: Remove the callbacks, replace with notifiers
  {
    std::shared_lock lk(cb_mu_);
    auto* maybe_ping = ping_item ? &*ping_item : nullptr;
    DVLOG(2) << "AddLogRecord: run callbacks for " << entry.ToString()
             << " num callbacks: " << change_cb_arr_.size();

    for (const auto& k_v : change_cb_arr_) {
      k_v.second(*item, await, maybe_ping);
    }
  }
}

uint32_t JournalSlice::RegisterOnChange(ChangeCallback cb) {
  lock_guard lk(cb_mu_);
  uint32_t id = next_cb_id_++;
  change_cb_arr_.emplace_back(id, std::move(cb));
  return id;
}

void JournalSlice::UnregisterOnChange(uint32_t id) {
  lock_guard lk(cb_mu_);
  auto it = find_if(change_cb_arr_.begin(), change_cb_arr_.end(),
                    [id](const auto& e) { return e.first == id; });
  CHECK(it != change_cb_arr_.end());
  change_cb_arr_.erase(it);
}

}  // namespace journal
}  // namespace dfly
