
// Copyright 2023, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/detail/save_stages_controller.h"

#include <absl/strings/match.h>

#include "base/flags.h"
#include "base/logging.h"
#include "server/detail/snapshot_storage.h"
#include "server/main_service.h"
#include "server/script_mgr.h"
#include "server/transaction.h"
#include "strings/human_readable.h"

using namespace std;

ABSL_DECLARE_FLAG(string, dir);
ABSL_DECLARE_FLAG(string, dbfilename);

namespace dfly {
namespace detail {

using namespace util;
using absl::GetFlag;
using absl::StrCat;
using fb2::OpenLinux;

namespace fs = std::filesystem;

namespace {

bool IsCloudPath(string_view path) {
  return absl::StartsWith(path, kS3Prefix);
}

// Create a directory and all its parents if they don't exist.
error_code CreateDirs(fs::path dir_path) {
  error_code ec;
  fs::file_status dir_status = fs::status(dir_path, ec);
  if (ec == errc::no_such_file_or_directory) {
    fs::create_directories(dir_path, ec);
    if (!ec)
      dir_status = fs::status(dir_path, ec);
  }
  return ec;
}

// modifies 'filename' to be "filename-postfix.extension"
void SetExtension(absl::AlphaNum postfix, string_view extension, fs::path* filename) {
  filename->replace_extension();  // clear if exists
  *filename += StrCat("-", postfix, extension);
}

void ExtendDfsFilenameWithShard(int shard, string_view extension, fs::path* filename) {
  // dragonfly snapshot.
  SetExtension(absl::Dec(shard, absl::kZeroPad4), extension, filename);
}

}  // namespace

GenericError ValidateFilename(const fs::path& filename, bool new_version) {
  if (filename.empty()) {
    return {};
  }

  string filename_str = filename.string();
  if (filename_str.front() == '"') {
    return {
        "filename should not start with '\"', could it be that you put quotes in the flagfile?"};
  }

  bool is_cloud_path = IsCloudPath(filename_str);

  if (!filename.parent_path().empty() && !is_cloud_path) {
    return {absl::StrCat("filename may not contain directory separators (Got \"", filename.c_str(),
                         "\"). dbfilename should specify the filename without the directory")};
  }

  if (!filename.has_extension()) {
    return {};
  }

  if (new_version) {
    if (absl::EqualsIgnoreCase(filename.extension().c_str(), ".rdb")) {
      return {absl::StrCat(
          "DF snapshot format is used but '.rdb' extension was given. Use --nodf_snapshot_format "
          "or remove the filename extension.")};
    } else {
      return {absl::StrCat("DF snapshot format requires no filename extension. Got \"",
                           filename.extension().c_str(), "\"")};
    }
  }
  if (!new_version && !absl::EqualsIgnoreCase(filename.extension().c_str(), ".rdb")) {
    return {absl::StrCat("Bad filename extension \"", filename.extension().c_str(),
                         "\" for SAVE with type RDB")};
  }
  return {};
}

GenericError RdbSnapshot::Start(SaveMode save_mode, const std::string& path,
                                const RdbSaver::GlobalData& glob_data) {
  VLOG(1) << "Saving RDB " << path;

  CHECK_NOTNULL(snapshot_storage_);
  auto res = snapshot_storage_->OpenWriteFile(path);
  if (!res) {
    return res.error();
  }

  auto [file, file_type] = *res;
  io_sink_.reset(file);

  is_linux_file_ = file_type & FileType::IO_URING;
  bool align_writes = (file_type & FileType::DIRECT) != 0;
  saver_.reset(new RdbSaver(io_sink_.get(), save_mode, align_writes));

  return saver_->SaveHeader(std::move(glob_data));
}

error_code RdbSnapshot::SaveBody() {
  return saver_->SaveBody(&cntx_, &freq_map_);
}

size_t RdbSnapshot::GetSaveBuffersSize() {
  CHECK(saver_);
  return saver_->GetTotalBuffersSize();
}

error_code RdbSnapshot::Close() {
  if (is_linux_file_) {
    return static_cast<LinuxWriteWrapper*>(io_sink_.get())->Close();
  }
  return static_cast<io::WriteFile*>(io_sink_.get())->Close();
}

void RdbSnapshot::StartInShard(EngineShard* shard) {
  saver_->StartSnapshotInShard(false, cntx_.GetCancellation(), shard);
  started_ = true;
}

SaveStagesController::SaveStagesController(SaveStagesInputs&& inputs)
    : SaveStagesInputs{std::move(inputs)} {
  start_time_ = absl::Now();
}

SaveStagesController::~SaveStagesController() {
}

SaveInfo SaveStagesController::Save() {
  if (auto err = BuildFullPath(); err) {
    shared_err_ = err;
    return GetSaveInfo();
  }

  InitResources();

  if (use_dfs_format_)
    SaveDfs();
  else
    SaveRdb();

  RunStage(&SaveStagesController::SaveCb);

  RunStage(&SaveStagesController::CloseCb);

  FinalizeFileMovement();

  return GetSaveInfo();
}

size_t SaveStagesController::GetSaveBuffersSize() {
  std::atomic<size_t> total_bytes{0};

  auto add_snapshot_bytes = [this, &total_bytes](ShardId sid) {
    if (auto& snapshot = snapshots_[sid].first; snapshot && snapshot->HasStarted()) {
      total_bytes.fetch_add(snapshot->GetSaveBuffersSize(), memory_order_relaxed);
    }
  };

  if (snapshots_.size() > 0) {
    if (use_dfs_format_) {
      shard_set->RunBriefInParallel([&](EngineShard* es) { add_snapshot_bytes(es->shard_id()); });

    } else {
      // When rdb format save is running, there is only one rdb saver instance, it is running on the
      // connection thread that runs the save command.
      add_snapshot_bytes(0);
    }
  }

  return total_bytes.load(memory_order_relaxed);
}

// In the new version (.dfs) we store a file for every shard and one more summary file.
// Summary file is always last in snapshots array.
void SaveStagesController::SaveDfs() {
  // Extend all filenames with -{sid} or -summary and append .dfs.tmp
  const string_view ext = is_cloud_ ? ".dfs" : ".dfs.tmp";
  ShardId sid = 0;
  for (auto& [_, filename] : snapshots_) {
    filename = full_path_;
    if (sid < shard_set->size())
      ExtendDfsFilenameWithShard(sid++, ext, &filename);
    else
      SetExtension("summary", ext, &filename);
  }

  // Save summary file.
  SaveDfsSingle(nullptr);

  // Save shard files.
  auto cb = [this](Transaction* t, EngineShard* shard) {
    SaveDfsSingle(shard);
    return OpStatus::OK;
  };
  trans_->ScheduleSingleHop(std::move(cb));
}

// Start saving a dfs file on shard
void SaveStagesController::SaveDfsSingle(EngineShard* shard) {
  // for summary file, shard=null and index=shard_set->size(), see SaveDfs() above
  auto& [snapshot, filename] = snapshots_[shard ? shard->shard_id() : shard_set->size()];

  SaveMode mode = shard == nullptr ? SaveMode::SUMMARY : SaveMode::SINGLE_SHARD;
  auto glob_data = shard == nullptr ? RdbSaver::GetGlobalData(service_) : RdbSaver::GlobalData{};

  if (auto err = snapshot->Start(mode, filename, glob_data); err) {
    shared_err_ = err;
    snapshot.reset();
    return;
  }

  if (mode == SaveMode::SINGLE_SHARD)
    snapshot->StartInShard(shard);
}

// Save a single rdb file
void SaveStagesController::SaveRdb() {
  auto& [snapshot, filename] = snapshots_.front();

  filename = full_path_;
  if (!filename.has_extension())
    filename += ".rdb";
  if (!is_cloud_)
    filename += ".tmp";

  if (auto err = snapshot->Start(SaveMode::RDB, filename, RdbSaver::GetGlobalData(service_)); err) {
    snapshot.reset();
    return;
  }

  auto cb = [snapshot = snapshot.get()](Transaction* t, EngineShard* shard) {
    snapshot->StartInShard(shard);
    return OpStatus::OK;
  };
  trans_->ScheduleSingleHop(std::move(cb));
}

uint32_t SaveStagesController::GetCurrentSaveDuration() {
  return (absl::Now() - start_time_) / absl::Seconds(1);
}

SaveInfo SaveStagesController::GetSaveInfo() {
  SaveInfo info;
  info.save_time = absl::ToUnixSeconds(start_time_);
  info.duration_sec = GetCurrentSaveDuration();

  if (shared_err_) {
    info.error = *shared_err_;
    return info;
  }

  fs::path resulting_path = full_path_;
  if (use_dfs_format_)
    SetExtension("summary", ".dfs", &resulting_path);
  else
    resulting_path.replace_extension();  // remove .tmp

  LOG(INFO) << "Saving " << resulting_path << " finished after "
            << strings::HumanReadableElapsedTime(info.duration_sec);

  info.freq_map.clear();
  for (const auto& k_v : rdb_name_map_) {
    info.freq_map.emplace_back(k_v);
  }

  info.file_name = resulting_path.generic_string();

  return info;
}

void SaveStagesController::InitResources() {
  snapshots_.resize(use_dfs_format_ ? shard_set->size() + 1 : 1);
  for (auto& [snapshot, _] : snapshots_)
    snapshot = make_unique<RdbSnapshot>(fq_threadpool_, snapshot_storage_.get());
}

// Remove .tmp extension or delete files in case of error
void SaveStagesController::FinalizeFileMovement() {
  if (is_cloud_)
    return;

  // If the shared_err is set, the snapshot saving failed
  bool has_error = bool(shared_err_);

  for (const auto& [_, filename] : snapshots_) {
    if (has_error)
      filesystem::remove(filename);
    else
      filesystem::rename(filename, fs::path{filename}.replace_extension(""));
  }
}

// Build full path: get dir, try creating dirs, get filename with placeholder
GenericError SaveStagesController::BuildFullPath() {
  fs::path dir_path = GetFlag(FLAGS_dir);
  if (!dir_path.empty()) {
    if (auto ec = CreateDirs(dir_path); ec)
      return {ec, "Failed to create directories"};
  }

  fs::path filename = basename_.empty() ? GetFlag(FLAGS_dbfilename) : basename_;
  if (filename.empty())
    return {"filename is not specified"};

  if (auto err = ValidateFilename(filename, use_dfs_format_); err)
    return err;

  SubstituteFilenamePlaceholders(
      &filename, {.ts = "%Y-%m-%dT%H:%M:%S", .year = "%Y", .month = "%m", .day = "%d"});
  filename = absl::FormatTime(filename.string(), start_time_, absl::LocalTimeZone());
  full_path_ = dir_path / filename;
  is_cloud_ = IsCloudPath(full_path_.string());
  return {};
}

void SaveStagesController::SaveCb(unsigned index) {
  if (auto& snapshot = snapshots_[index].first; snapshot && snapshot->HasStarted())
    shared_err_ = snapshot->SaveBody();
}

void SaveStagesController::CloseCb(unsigned index) {
  if (auto& snapshot = snapshots_[index].first; snapshot) {
    shared_err_ = snapshot->Close();

    unique_lock lk{rdb_name_map_mu_};
    for (const auto& k_v : snapshot->freq_map())
      rdb_name_map_[RdbTypeName(k_v.first)] += k_v.second;
    lk.unlock();
    snapshot.reset();
  }

  if (auto* es = EngineShard::tlocal(); use_dfs_format_ && es)
    es->db_slice().ResetUpdateEvents();
}

void SaveStagesController::RunStage(void (SaveStagesController::*cb)(unsigned)) {
  if (use_dfs_format_) {
    shard_set->RunBlockingInParallel([&](EngineShard* es) { (this->*cb)(es->shard_id()); });
    (this->*cb)(shard_set->size());
  } else {
    (this->*cb)(0);
  }
}

}  // namespace detail
}  // namespace dfly
