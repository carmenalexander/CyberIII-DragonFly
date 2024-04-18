// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//
#include "facade/reply_builder.h"

#include <absl/cleanup/cleanup.h>
#include <absl/container/fixed_array.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <double-conversion/double-to-string.h>

#include "absl/strings/escaping.h"
#include "base/logging.h"
#include "core/heap_size.h"
#include "facade/error.h"
#include "util/fibers/proactor_base.h"

using namespace std;
using absl::StrAppend;
using namespace double_conversion;

namespace facade {

namespace {

inline iovec constexpr IoVec(std::string_view s) {
  iovec r{const_cast<char*>(s.data()), s.size()};
  return r;
}

constexpr char kCRLF[] = "\r\n";
constexpr char kErrPref[] = "-ERR ";
constexpr char kSimplePref[] = "+";

constexpr unsigned kConvFlags =
    DoubleToStringConverter::UNIQUE_ZERO | DoubleToStringConverter::EMIT_POSITIVE_EXPONENT_SIGN;

DoubleToStringConverter dfly_conv(kConvFlags, "inf", "nan", 'e', -6, 21, 6, 0);

const char* NullString(bool resp3) {
  return resp3 ? "_\r\n" : "$-1\r\n";
}

}  // namespace

SinkReplyBuilder::MGetResponse::~MGetResponse() {
  while (storage_list) {
    auto* next = storage_list->next;
    delete[] reinterpret_cast<char*>(storage_list);
    storage_list = next;
  }
}

SinkReplyBuilder::SinkReplyBuilder(::io::Sink* sink)
    : sink_(sink),
      should_batch_(false),
      should_aggregate_(false),
      has_replied_(true),
      send_active_(false) {
}

void SinkReplyBuilder::CloseConnection() {
  if (!ec_)
    ec_ = std::make_error_code(std::errc::connection_aborted);
}

void SinkReplyBuilder::ResetThreadLocalStats() {
  tl_facade_stats->reply_stats = {};
}

void SinkReplyBuilder::Send(const iovec* v, uint32_t len) {
  has_replied_ = true;
  DCHECK(sink_);
  constexpr size_t kMaxBatchSize = 1024;

  size_t bsize = 0;
  for (unsigned i = 0; i < len; ++i) {
    bsize += v[i].iov_len;
  }

  // Allow batching with up to kMaxBatchSize of data.
  if ((should_batch_ || should_aggregate_) && (batch_.size() + bsize < kMaxBatchSize)) {
    batch_.reserve(batch_.size() + bsize);
    for (unsigned i = 0; i < len; ++i) {
      std::string_view src((char*)v[i].iov_base, v[i].iov_len);
      DVLOG(3) << "Appending to stream " << absl::CHexEscape(src);
      batch_.append(src.data(), src.size());
    }
    DVLOG(2) << "Batched " << bsize << " bytes";
    return;
  }

  int64_t before_ns = util::fb2::ProactorBase::GetMonotonicTimeNs();
  error_code ec;
  send_active_ = true;
  tl_facade_stats->reply_stats.io_write_cnt++;
  tl_facade_stats->reply_stats.io_write_bytes += bsize;
  DVLOG(2) << "Writing " << bsize << " bytes of len " << len;

  if (batch_.empty()) {
    ec = sink_->Write(v, len);
  } else {
    DVLOG(3) << "Sending batch to stream :" << absl::CHexEscape(batch_);

    tl_facade_stats->reply_stats.io_write_bytes += batch_.size();

    iovec tmp[len + 1];
    tmp[0].iov_base = batch_.data();
    tmp[0].iov_len = batch_.size();
    copy(v, v + len, tmp + 1);
    ec = sink_->Write(tmp, len + 1);
    batch_.clear();
  }
  send_active_ = false;
  int64_t after_ns = util::fb2::ProactorBase::GetMonotonicTimeNs();
  tl_facade_stats->reply_stats.send_stats.count++;
  tl_facade_stats->reply_stats.send_stats.total_duration += (after_ns - before_ns) / 1'000;

  if (ec) {
    DVLOG(1) << "Error writing to stream: " << ec.message();
    ec_ = ec;
  }
}

void SinkReplyBuilder::SendRaw(std::string_view raw) {
  iovec v = {IoVec(raw)};

  Send(&v, 1);
}

void SinkReplyBuilder::ExpectReply() {
  has_replied_ = false;
}

bool SinkReplyBuilder::HasReplied() const {
  return has_replied_;
}

void SinkReplyBuilder::SendError(ErrorReply error) {
  if (error.status)
    return SendError(*error.status);

  string_view message_sv = visit([](auto&& str) -> string_view { return str; }, error.message);
  SendError(message_sv, error.kind);
}

void SinkReplyBuilder::SendError(OpStatus status) {
  if (status == OpStatus::OK) {
    SendOk();
  } else {
    SendError(StatusToMsg(status));
  }
}

void SinkReplyBuilder::SendRawVec(absl::Span<const std::string_view> msg_vec) {
  absl::FixedArray<iovec, 16> arr(msg_vec.size());

  for (unsigned i = 0; i < msg_vec.size(); ++i) {
    arr[i].iov_base = const_cast<char*>(msg_vec[i].data());
    arr[i].iov_len = msg_vec[i].size();
  }
  Send(arr.data(), msg_vec.size());
}

void SinkReplyBuilder::StartAggregate() {
  DVLOG(1) << "StartAggregate";
  should_aggregate_ = true;
}

void SinkReplyBuilder::StopAggregate() {
  DVLOG(1) << "StopAggregate";
  should_aggregate_ = false;

  if (should_batch_)
    return;

  FlushBatch();
}

void SinkReplyBuilder::SetBatchMode(bool batch) {
  DVLOG(1) << "SetBatchMode(" << (batch ? "true" : "false") << ")";
  should_batch_ = batch;
}

void SinkReplyBuilder::FlushBatch() {
  if (batch_.empty())
    return;

  error_code ec = sink_->Write(io::Buffer(batch_));
  batch_.clear();
  if (ec) {
    DVLOG(1) << "Error flushing to stream: " << ec.message();
    ec_ = ec;
  }
}

size_t SinkReplyBuilder::UsedMemory() const {
  return dfly::HeapSize(batch_);
}

MCReplyBuilder::MCReplyBuilder(::io::Sink* sink) : SinkReplyBuilder(sink), noreply_(false) {
}

void MCReplyBuilder::SendSimpleString(std::string_view str) {
  if (noreply_)
    return;

  iovec v[2] = {IoVec(str), IoVec(kCRLF)};

  Send(v, ABSL_ARRAYSIZE(v));
}

void MCReplyBuilder::SendStored() {
  SendSimpleString("STORED");
}

void MCReplyBuilder::SendLong(long val) {
  char buf[32];
  char* next = absl::numbers_internal::FastIntToBuffer(val, buf);
  SendSimpleString(string_view(buf, next - buf));
}

void MCReplyBuilder::SendMGetResponse(MGetResponse resp) {
  string header;
  for (unsigned i = 0; i < resp.resp_arr.size(); ++i) {
    if (resp.resp_arr[i]) {
      const auto& src = *resp.resp_arr[i];
      absl::StrAppend(&header, "VALUE ", src.key, " ", src.mc_flag, " ", src.value.size());
      if (src.mc_ver) {
        absl::StrAppend(&header, " ", src.mc_ver);
      }

      absl::StrAppend(&header, "\r\n");
      iovec v[] = {IoVec(header), IoVec(src.value), IoVec(kCRLF)};
      Send(v, ABSL_ARRAYSIZE(v));
      header.clear();
    }
  }
  SendSimpleString("END");
}

void MCReplyBuilder::SendError(string_view str, std::string_view type) {
  SendSimpleString(absl::StrCat("SERVER_ERROR ", str));
}

void MCReplyBuilder::SendProtocolError(std::string_view str) {
  SendSimpleString(absl::StrCat("CLIENT_ERROR ", str));
}

bool MCReplyBuilder::NoReply() const {
  return noreply_;
}

void MCReplyBuilder::SendClientError(string_view str) {
  iovec v[] = {IoVec("CLIENT_ERROR "), IoVec(str), IoVec(kCRLF)};
  Send(v, ABSL_ARRAYSIZE(v));
}

void MCReplyBuilder::SendSetSkipped() {
  SendSimpleString("NOT_STORED");
}

void MCReplyBuilder::SendNotFound() {
  SendSimpleString("NOT_FOUND");
}

size_t RedisReplyBuilder::WrappedStrSpan::Size() const {
  return visit([](auto arr) { return arr.size(); }, (const StrSpan&)*this);
}

string_view RedisReplyBuilder::WrappedStrSpan::operator[](size_t i) const {
  return visit([i](auto arr) { return string_view{arr[i]}; }, (const StrSpan&)*this);
}

char* RedisReplyBuilder::FormatDouble(double val, char* dest, unsigned dest_len) {
  StringBuilder sb(dest, dest_len);
  CHECK(dfly_conv.ToShortest(val, &sb));
  return sb.Finalize();
}

RedisReplyBuilder::RedisReplyBuilder(::io::Sink* sink) : SinkReplyBuilder(sink) {
}

void RedisReplyBuilder::SetResp3(bool is_resp3) {
  is_resp3_ = is_resp3;
}

bool RedisReplyBuilder::IsResp3() const {
  return is_resp3_;
}

void RedisReplyBuilder::SendError(string_view str, string_view err_type) {
  VLOG(1) << "Error: " << str;

  if (err_type.empty()) {
    err_type = str;
    if (err_type == kSyntaxErr)
      err_type = kSyntaxErrType;
  }

  tl_facade_stats->reply_stats.err_count[err_type]++;

  if (str[0] == '-') {
    iovec v[] = {IoVec(str), IoVec(kCRLF)};
    Send(v, ABSL_ARRAYSIZE(v));
    return;
  }

  iovec v[] = {IoVec(kErrPref), IoVec(str), IoVec(kCRLF)};
  Send(v, ABSL_ARRAYSIZE(v));
}

void RedisReplyBuilder::SendProtocolError(std::string_view str) {
  SendError(absl::StrCat("-ERR Protocol error: ", str), "protocol_error");
}

void RedisReplyBuilder::SendSimpleString(std::string_view str) {
  iovec v[3] = {IoVec(kSimplePref), IoVec(str), IoVec(kCRLF)};

  Send(v, ABSL_ARRAYSIZE(v));
}

void RedisReplyBuilder::SendStored() {
  SendSimpleString("OK");
}

void RedisReplyBuilder::SendSetSkipped() {
  SendNull();
}

void RedisReplyBuilder::SendNull() {
  iovec v[] = {IoVec(NullString(is_resp3_))};

  Send(v, ABSL_ARRAYSIZE(v));
}

void RedisReplyBuilder::SendBulkString(std::string_view str) {
  char tmp[absl::numbers_internal::kFastToBufferSize + 3];
  tmp[0] = '$';  // Format length
  char* next = absl::numbers_internal::FastIntToBuffer(uint32_t(str.size()), tmp + 1);
  *next++ = '\r';
  *next++ = '\n';

  std::string_view lenpref{tmp, size_t(next - tmp)};

  // 3 parts: length, string and CRLF.
  iovec v[3] = {IoVec(lenpref), IoVec(str), IoVec(kCRLF)};

  return Send(v, ABSL_ARRAYSIZE(v));
}

void RedisReplyBuilder::SendVerbatimString(std::string_view str, VerbatimFormat format) {
  if (!is_resp3_)
    return SendBulkString(str);

  char tmp[absl::numbers_internal::kFastToBufferSize + 7];
  tmp[0] = '=';
  // + 4 because format is three byte, and need to be followed by a ":"
  char* next = absl::numbers_internal::FastIntToBuffer(uint32_t(str.size() + 4), tmp + 1);
  *next++ = '\r';
  *next++ = '\n';

  DCHECK(format <= VerbatimFormat::MARKDOWN);
  if (format == VerbatimFormat::TXT)
    strcpy(next, "txt:");
  else if (format == VerbatimFormat::MARKDOWN)
    strcpy(next, "mkd:");
  next += 4;
  std::string_view lenpref{tmp, size_t(next - tmp)};
  iovec v[3] = {IoVec(lenpref), IoVec(str), IoVec(kCRLF)};
  return Send(v, ABSL_ARRAYSIZE(v));
}

void RedisReplyBuilder::SendLong(long num) {
  string str = absl::StrCat(":", num, kCRLF);
  SendRaw(str);
}

void RedisReplyBuilder::SendScoredArray(const std::vector<std::pair<std::string, double>>& arr,
                                        bool with_scores) {
  ReplyAggregator agg(this);
  if (!with_scores) {
    auto cb = [&](size_t indx) -> string_view { return arr[indx].first; };

    SendStringArrInternal(arr.size(), std::move(cb), CollectionType::ARRAY);
    return;
  }

  // DoubleToStringConverter::kBase10MaximalLength is 17.
  char buf[64];

  if (!is_resp3_) {  // RESP2 formats withscores as a flat array.
    auto cb = [&](size_t indx) -> string_view {
      if (indx % 2 == 0)
        return arr[indx / 2].first;

      // NOTE: we reuse the same buffer, assuming that SendStringArrInternal does not reference
      // previous string_views. The assumption holds for small strings like
      // doubles because SendStringArrInternal employs small string optimization.
      // It's a bit hacky but saves allocations.
      return FormatDouble(arr[indx / 2].second, buf, sizeof(buf));
    };

    SendStringArrInternal(arr.size() * 2, std::move(cb), CollectionType::ARRAY);
    return;
  }

  // Resp3 formats withscores as array of (key, score) pairs.
  // TODO: to implement efficient serializing by extending SendStringArrInternal to support
  // 2-level arrays.
  StartArray(arr.size());
  for (const auto& p : arr) {
    StartArray(2);
    SendBulkString(p.first);
    SendDouble(p.second);
  }
}

void RedisReplyBuilder::SendDouble(double val) {
  char buf[64];

  char* start = FormatDouble(val, buf, sizeof(buf));

  if (!is_resp3_) {
    SendBulkString(start);
  } else {
    // RESP3
    SendRaw(absl::StrCat(",", start, kCRLF));
  }
}

void RedisReplyBuilder::SendMGetResponse(MGetResponse resp) {
  DCHECK(!resp.resp_arr.empty());

  size_t size = resp.resp_arr.size();

  size_t vec_len = std::min<size_t>(32, size);

  constexpr size_t kBatchLen = 32 * 2 + 2;  // (blob_size, blob) * 32 + 2 spares
  iovec vec_batch[kBatchLen];

  // for all the meta data to fill the vec batch. 10 digits for the blob size and 6 for
  // $, \r, \n, \r, \n
  absl::FixedArray<char, 64> meta((vec_len + 2) * 16);  // 2 for header and next item meta data.

  char* next = meta.data();
  char* cur_meta = next;
  *next++ = '*';
  next = absl::numbers_internal::FastIntToBuffer(size, next);
  *next++ = '\r';
  *next++ = '\n';

  unsigned vec_indx = 0;
  const char* nullstr = NullString(is_resp3_);
  size_t nulllen = strlen(nullstr);
  auto get_pending_metabuf = [&] { return string_view{cur_meta, size_t(next - cur_meta)}; };

  for (unsigned i = 0; i < size; ++i) {
    DCHECK_GE(meta.end() - next, 16);  // We have at least 16 bytes for the meta data.
    if (resp.resp_arr[i]) {
      string_view blob = resp.resp_arr[i]->value;

      *next++ = '$';
      next = absl::numbers_internal::FastIntToBuffer(blob.size(), next);
      *next++ = '\r';
      *next++ = '\n';
      DCHECK_GT(next - cur_meta, 0);

      vec_batch[vec_indx++] = IoVec(get_pending_metabuf());
      vec_batch[vec_indx++] = IoVec(blob);
      cur_meta = next;  // we combine the CRLF with the next item meta data.
      *next++ = '\r';
      *next++ = '\n';
    } else {
      memcpy(next, nullstr, nulllen);
      next += nulllen;
    }

    if (vec_indx >= (kBatchLen - 2) || (meta.end() - next < 16)) {
      // we have space for at least one iovec because in the worst case we reached (kBatchLen - 3)
      // and then filled 2 vectors in the previous iteration.
      DCHECK_LE(vec_indx, kBatchLen - 1);

      // if we do not have enough space in the meta buffer, we add the meta data to the
      // vector batch and reset it.
      if (meta.end() - next < 16) {
        vec_batch[vec_indx++] = IoVec(get_pending_metabuf());
        next = meta.data();
        cur_meta = next;
      }

      Send(vec_batch, vec_indx);
      if (ec_)
        return;

      vec_indx = 0;
      size_t meta_len = next - cur_meta;
      memcpy(meta.data(), cur_meta, meta_len);
      cur_meta = meta.data();
      next = cur_meta + meta_len;
    }
  }

  if (next - cur_meta > 0) {
    vec_batch[vec_indx++] = IoVec(get_pending_metabuf());
  }
  if (vec_indx > 0)
    Send(vec_batch, vec_indx);
}

void RedisReplyBuilder::SendSimpleStrArr(StrSpan arr) {
  WrappedStrSpan warr{arr};

  string res = absl::StrCat("*", warr.Size(), kCRLF);

  for (unsigned i = 0; i < warr.Size(); i++)
    StrAppend(&res, "+", warr[i], kCRLF);

  SendRaw(res);
}

void RedisReplyBuilder::SendNullArray() {
  SendRaw("*-1\r\n");
}

void RedisReplyBuilder::SendEmptyArray() {
  StartArray(0);
}

void RedisReplyBuilder::SendStringArr(StrSpan arr, CollectionType type) {
  WrappedStrSpan warr{arr};

  if (type == ARRAY && warr.Size() == 0) {
    SendRaw("*0\r\n");
    return;
  }

  auto cb = [&](size_t i) { return warr[i]; };

  SendStringArrInternal(warr.Size(), std::move(cb), type);
}

void RedisReplyBuilder::StartArray(unsigned len) {
  StartCollection(len, ARRAY);
}

constexpr static string_view START_SYMBOLS[] = {"*", "~", "%", ">"};
static_assert(START_SYMBOLS[RedisReplyBuilder::MAP] == "%" &&
              START_SYMBOLS[RedisReplyBuilder::SET] == "~");

void RedisReplyBuilder::StartCollection(unsigned len, CollectionType type) {
  if (!is_resp3_) {  // Flatten for Resp2
    if (type == MAP)
      len *= 2;
    type = ARRAY;
  }

  DVLOG(2) << "StartCollection(" << len << ", " << type << ")";

  // We do not want to send multiple packets for small responses because these
  // trigger TCP-related artifacts (e.g. Nagle's algorithm) that slow down the delivery of the whole
  // response.
  bool prev = should_aggregate_;
  should_aggregate_ |= (len > 0);
  SendRaw(absl::StrCat(START_SYMBOLS[type], len, kCRLF));
  should_aggregate_ = prev;
}

// This implementation a bit complicated because it uses vectorized
// send to send an array. The problem with that is the OS limits vector length
// to low numbers (around 1024). Therefore, to make it robust we send the array in batches.
// We limit the vector length to 256 and when it fills up we flush it to the socket and continue
// iterating.
void RedisReplyBuilder::SendStringArrInternal(
    size_t size, absl::FunctionRef<std::string_view(unsigned)> producer, CollectionType type) {
  size_t header_len = size;
  string_view type_char = "*";
  if (is_resp3_) {
    type_char = START_SYMBOLS[type];
    if (type == MAP)
      header_len /= 2;  // Each key value pair counts as one.
  }

  if (header_len == 0) {
    SendRaw(absl::StrCat(type_char, "0\r\n"));
    return;
  }

  // When vector length is too long, Send returns EMSGSIZE.
  size_t vec_len = std::min<size_t>(124u, size);

  absl::FixedArray<iovec, 16> vec(vec_len * 2 + 2);
  absl::FixedArray<char, 128> meta(vec_len * 32 + 64);  // 32 bytes per element + spare space

  char* next = meta.data();

  auto serialize_len = [&](char prefix, size_t len) {
    *next++ = prefix;
    next = absl::numbers_internal::FastIntToBuffer(len, next);
    *next++ = '\r';
    *next++ = '\n';
  };

  serialize_len(type_char[0], header_len);
  vec[0] = IoVec(string_view{meta.data(), size_t(next - meta.data())});
  char* start = next;

  unsigned vec_indx = 1;
  string_view src;
  for (unsigned i = 0; i < size; ++i) {
    src = producer(i);
    serialize_len('$', src.size());

    // add serialized len blob
    vec[vec_indx++] = IoVec(string_view{start, size_t(next - start)});
    DCHECK_GT(next - start, 0);

    start = next;

    // copy data either by referencing via an iovec or copying inline into meta buf.
    if (src.size() >= 30) {
      vec[vec_indx++] = IoVec(src);
    } else if (src.size() > 0) {
      memcpy(next, src.data(), src.size());
      vec[vec_indx - 1].iov_len += src.size();  // extend the reference
      next += src.size();
      start = next;
    }
    *next++ = '\r';
    *next++ = '\n';

    // we keep at least 40 bytes to have enough place for a small string as well as its length.
    if (vec_indx + 1 >= vec.size() || (meta.end() - next < 40)) {
      // Flush the iovec array.
      if (i < size - 1 || vec_indx == vec.size()) {
        Send(vec.data(), vec_indx);
        if (ec_)
          return;

        vec_indx = 0;
        start = meta.data();
        next = start + 2;
        start[0] = '\r';
        start[1] = '\n';
      }
    }
  }

  vec[vec_indx].iov_base = start;
  vec[vec_indx].iov_len = 2;
  Send(vec.data(), vec_indx + 1);
}

void ReqSerializer::SendCommand(std::string_view str) {
  VLOG(2) << "SendCommand: " << str;

  iovec v[] = {IoVec(str), IoVec(kCRLF)};
  ec_ = sink_->Write(v, ABSL_ARRAYSIZE(v));
}

}  // namespace facade
