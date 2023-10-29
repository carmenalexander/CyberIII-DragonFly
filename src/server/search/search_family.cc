// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "server/search/search_family.h"

#include <absl/container/flat_hash_map.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>

#include <atomic>
#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>
#include <variant>
#include <vector>

#include "base/logging.h"
#include "core/json_object.h"
#include "core/search/search.h"
#include "core/search/vector_utils.h"
#include "facade/cmd_arg_parser.h"
#include "facade/error.h"
#include "facade/reply_builder.h"
#include "server/acl/acl_commands_def.h"
#include "server/command_registry.h"
#include "server/conn_context.h"
#include "server/container_utils.h"
#include "server/engine_shard_set.h"
#include "server/search/doc_index.h"
#include "server/transaction.h"

namespace dfly {

using namespace std;
using namespace facade;

namespace {

static const set<string_view> kIgnoredOptions = {"WEIGHT", "SEPARATOR"};

bool IsValidJsonPath(string_view path) {
  error_code ec;
  jsoncons::jsonpath::make_expression<JsonType>(path, ec);
  return !ec;
}

search::SchemaField::VectorParams ParseVectorParams(CmdArgParser* parser) {
  size_t dim = 0;
  auto sim = search::VectorSimilarity::L2;
  size_t capacity = 1000;

  bool use_hnsw = parser->ToUpper().Next().Case("HNSW", true).Case("FLAT", false);
  size_t num_args = parser->Next().Int<size_t>();

  for (size_t i = 0; i * 2 < num_args; i++) {
    parser->ToUpper();

    if (parser->Check("DIM").ExpectTail(1)) {
      dim = parser->Next().Int<size_t>();
      continue;
    }

    if (parser->Check("DISTANCE_METRIC").ExpectTail(1)) {
      sim = parser->Next()
                .Case("L2", search::VectorSimilarity::L2)
                .Case("COSINE", search::VectorSimilarity::COSINE);
      continue;
    }

    if (parser->Check("INITIAL_CAP").ExpectTail(1)) {
      capacity = parser->Next().Int<size_t>();
      continue;
    }

    parser->Skip(2);
  }

  return {use_hnsw, dim, sim, capacity};
}

optional<search::Schema> ParseSchemaOrReply(DocIndex::DataType type, CmdArgParser parser,
                                            ConnectionContext* cntx) {
  search::Schema schema;

  while (parser.HasNext()) {
    string_view field = parser.Next();
    string_view field_alias = field;

    // Verify json path is correct
    if (type == DocIndex::JSON && !IsValidJsonPath(field)) {
      (*cntx)->SendError("Bad json path: " + string{field});
      return nullopt;
    }

    parser.ToUpper();

    // AS [alias]
    if (parser.Check("AS").ExpectTail(1).NextUpper())
      field_alias = parser.Next();

    // Determine type
    string_view type_str = parser.Next();
    auto type = ParseSearchFieldType(type_str);
    if (!type) {
      (*cntx)->SendError("Invalid field type: " + string{type_str});
      return nullopt;
    }

    // Vector fields include: {algorithm} num_args args...
    search::SchemaField::ParamsVariant params = std::monostate{};
    if (*type == search::SchemaField::VECTOR) {
      auto vector_params = ParseVectorParams(&parser);
      if (!parser.HasError() && vector_params.dim == 0) {
        (*cntx)->SendError("Knn vector dimension cannot be zero");
        return nullopt;
      }
      params = std::move(vector_params);
    }

    // Flags: check for SORTABLE and NOINDEX
    uint8_t flags = 0;
    while (parser.HasNext()) {
      if (parser.Check("NOINDEX").IgnoreCase()) {
        flags |= search::SchemaField::NOINDEX;
        continue;
      }

      if (parser.Check("SORTABLE").IgnoreCase()) {
        flags |= search::SchemaField::SORTABLE;
        continue;
      }

      break;
    }

    // Skip all trailing ignored parameters
    while (kIgnoredOptions.count(parser.Peek()) > 0)
      parser.Skip(2);

    schema.fields[field] = {*type, flags, string{field_alias}, std::move(params)};
  }

  // Build field name mapping table
  for (const auto& [field_ident, field_info] : schema.fields)
    schema.field_names[field_info.short_name] = field_ident;

  if (auto err = parser.Error(); err) {
    (*cntx)->SendError(err->MakeReply());
    return nullopt;
  }

  return schema;
}

optional<SearchParams> ParseSearchParamsOrReply(CmdArgParser parser, ConnectionContext* cntx) {
  SearchParams params;

  while (parser.ToUpper().HasNext()) {
    // [LIMIT offset total]
    if (parser.Check("LIMIT").ExpectTail(2)) {
      params.limit_offset = parser.Next().Int<size_t>();
      params.limit_total = parser.Next().Int<size_t>();
      continue;
    }

    // RETURN {num} [{ident} AS {name}...]
    if (parser.Check("RETURN").ExpectTail(1)) {
      size_t num_fields = parser.Next().Int<size_t>();
      params.return_fields = SearchParams::FieldReturnList{};
      while (params.return_fields->size() < num_fields) {
        string_view ident = parser.Next();
        string_view alias = parser.Check("AS").IgnoreCase().ExpectTail(1) ? parser.Next() : ident;
        params.return_fields->emplace_back(ident, alias);
      }
      continue;
    }

    // NOCONTENT
    if (parser.Check("NOCONTENT")) {
      params.return_fields = SearchParams::FieldReturnList{};
      continue;
    }

    // [PARAMS num(ignored) name(ignored) knn_vector]
    if (parser.Check("PARAMS").ExpectTail(1)) {
      size_t num_args = parser.Next().Int<size_t>();
      while (parser.HasNext() && params.query_params.Size() * 2 < num_args) {
        string_view k = parser.Next();
        string_view v = parser.Next();
        params.query_params[k] = v;
      }
      continue;
    }

    if (parser.Check("SORTBY").ExpectTail(1)) {
      params.sort_option =
          search::SortOption{string{parser.Next()}, bool(parser.Check("DESC").IgnoreCase())};
      continue;
    }

    // Unsupported parameters are ignored for now
    parser.Skip(1);
  }

  if (auto err = parser.Error(); err) {
    (*cntx)->SendError(err->MakeReply());
    return nullopt;
  }

  return params;
}

void SendSerializedDoc(const DocResult::SerializedValue& value, ConnectionContext* cntx) {
  (*cntx)->SendBulkString(value.key);
  (*cntx)->StartCollection(value.values.size(), RedisReplyBuilder::MAP);
  for (const auto& [k, v] : value.values) {
    (*cntx)->SendBulkString(k);
    (*cntx)->SendBulkString(v);
  }
}

struct MultishardSearch {
  MultishardSearch(ConnectionContext* cntx, std::string_view index_name,
                   search::SearchAlgorithm* search_algo, SearchParams params)
      : cntx_{cntx},
        index_name_{index_name},
        search_algo_{search_algo},
        params_{std::move(params)} {
    sharded_results_.resize(shard_set->size());
    if (search_algo_->IsProfilingEnabled())
      sharded_times_.resize(shard_set->size());
  }

  void RunAndReply() {
    // First, run search with probabilistic optimizations enabled.
    // If the result set was collected successfuly, reply.
    {
      params_.enable_cutoff = true;

      if (auto err = RunSearch(); err)
        return (*cntx_)->SendError(std::move(*err));

      auto incomplete_shards = BuildOrder();
      if (incomplete_shards.empty())
        return Reply();
    }

    VLOG(1) << "Failed completness check, refilling";

    // Otherwise, some results made it into the result set but were not serialized.
    // Try refilling the requested values. If no reordering occured, reply immediately, otherwise
    // try building a new order and reply if it is valid.
    {
      params_.enable_cutoff = false;

      auto refill_res = RunRefill();
      if (!refill_res.has_value())
        return (*cntx_)->SendError(std::move(refill_res.error()));

      if (bool no_reordering = refill_res.value(); no_reordering)
        return Reply();

      if (auto incomplete_shards = BuildOrder(); incomplete_shards.empty())
        return Reply();
    }

    VLOG(1) << "Failed refill and rebuild, re-searching";

    // At this step all optimizations failed. Run search without any cutoffs.
    {
      DCHECK(!params_.enable_cutoff);

      if (auto err = RunSearch(); err)
        return (*cntx_)->SendError(std::move(*err));

      auto incomplete_shards = BuildOrder();
      DCHECK(incomplete_shards.empty());
      Reply();
    }
  }

  struct ProfileInfo {
    size_t total = 0;
    size_t serialized = 0;
    size_t cutoff = 0;
    size_t hops = 0;
    std::vector<pair<search::AlgorithmProfile, absl::Duration>> profiles;
  };

  ProfileInfo GetProfileInfo() {
    ProfileInfo info;
    info.hops = hops_;

    for (size_t i = 0; i < sharded_results_.size(); i++) {
      const auto& sd = sharded_results_[i];
      size_t serialized = count_if(sd.docs.begin(), sd.docs.end(), [](const auto& doc_res) {
        return holds_alternative<DocResult::SerializedValue>(doc_res.value);
      });

      info.total += sd.total_hits;
      info.serialized += serialized;
      info.cutoff += sd.docs.size() - serialized;

      DCHECK(sd.profile);
      info.profiles.push_back({std::move(*sd.profile), sharded_times_[i]});
    }

    return info;
  }

 private:
  void Reply() {
    size_t total_count = 0;
    for (const auto& shard_docs : sharded_results_)
      total_count += shard_docs.total_hits;

    auto agg_info = search_algo_->HasAggregation();
    if (agg_info && agg_info->limit)
      total_count = min(total_count, *agg_info->limit);

    if (agg_info && !params_.ShouldReturnField(agg_info->alias))
      agg_info->alias = ""sv;

    size_t result_count =
        min(total_count - min(total_count, params_.limit_offset), params_.limit_total);

    facade::SinkReplyBuilder::ReplyAggregator agg{cntx_->reply_builder()};

    bool ids_only = params_.IdsOnly();
    size_t reply_size = ids_only ? (result_count + 1) : (result_count * 2 + 1);

    (*cntx_)->StartArray(reply_size);
    (*cntx_)->SendLong(total_count);

    for (size_t i = params_.limit_offset; i < ordered_docs_.size(); i++) {
      auto& value = get<DocResult::SerializedValue>(ordered_docs_[i]->value);
      if (ids_only) {
        (*cntx_)->SendBulkString(value.key);
        continue;
      }

      if (agg_info && !agg_info->alias.empty())
        value.values[agg_info->alias] = absl::StrCat(get<float>(ordered_docs_[i]->score));

      SendSerializedDoc(value, cntx_);
    }
  }

  // Run function f on all search indices, return first error
  std::optional<facade::ErrorReply> RunHandler(
      std::function<std::optional<ErrorReply>(EngineShard*, ShardDocIndex*)> f) {
    hops_++;
    AggregateValue<optional<facade::ErrorReply>> err;
    cntx_->transaction->ScheduleSingleHop([&](Transaction* t, EngineShard* es) {
      optional<absl::Time> start;
      if (search_algo_->IsProfilingEnabled())
        start = absl::Now();

      if (auto* index = es->search_indices()->GetIndex(index_name_); index)
        err = f(es, index);
      else
        err = facade::ErrorReply(string{index_name_} + ": no such index");

      if (start.has_value())
        sharded_times_[es->shard_id()] += (absl::Now() - *start);

      return OpStatus::OK;
    });
    return *err;
  }

  optional<facade::ErrorReply> RunSearch() {
    cntx_->transaction->Refurbish();

    return RunHandler([this](EngineShard* es, ShardDocIndex* index) -> optional<ErrorReply> {
      auto res = index->Search(cntx_->transaction->GetOpArgs(es), params_, search_algo_);
      if (!res.has_value())
        return std::move(res.error());
      sharded_results_[es->shard_id()] = std::move(res.value());
      return nullopt;
    });
  }

  io::Result<bool, facade::ErrorReply> RunRefill() {
    cntx_->transaction->Refurbish();

    atomic_uint failed_refills = 0;
    auto err = RunHandler([this, &failed_refills](EngineShard* es, ShardDocIndex* index) {
      bool refilled = index->Refill(cntx_->transaction->GetOpArgs(es), params_, search_algo_,
                                    &sharded_results_[es->shard_id()]);
      if (!refilled)
        failed_refills.fetch_add(1u);
      return nullopt;
    });

    if (err)
      return nonstd::make_unexpected(std::move(*err));
    return failed_refills == 0;
  }

  // Build order from results collected from shards
  absl::flat_hash_set<ShardId> BuildOrder() {
    ordered_docs_.clear();
    if (auto agg = search_algo_->HasAggregation(); agg) {
      BuildSortedOrder(*agg);
    } else {
      BuildLinearOrder();
    }
    return VerifyOrderCompletness();
  }

  void BuildLinearOrder() {
    size_t required = params_.limit_offset + params_.limit_total;

    for (size_t idx = 0;; idx++) {
      bool added = false;
      for (auto& shard_result : sharded_results_) {
        if (idx < shard_result.docs.size() && ordered_docs_.size() < required) {
          ordered_docs_.push_back(&shard_result.docs[idx]);
          added = true;
        }
      }
      if (!added)
        return;
    }
  }

  void BuildSortedOrder(search::AggregationInfo agg) {
    for (auto& shard_result : sharded_results_) {
      for (auto& doc : shard_result.docs) {
        ordered_docs_.push_back(&doc);
      }
    }

    size_t agg_limit = agg.limit.value_or(ordered_docs_.size());
    size_t prefix = min(params_.limit_offset + params_.limit_total, agg_limit);

    partial_sort(ordered_docs_.begin(), ordered_docs_.begin() + min(ordered_docs_.size(), prefix),
                 ordered_docs_.end(), [desc = agg.descending](const auto* l, const auto* r) {
                   return desc ? (l->score >= r->score) : (l->score < r->score);
                 });

    ordered_docs_.resize(min(ordered_docs_.size(), prefix));
  }

  absl::flat_hash_set<ShardId> VerifyOrderCompletness() {
    absl::flat_hash_set<ShardId> incomplete_shards;
    for (auto* doc : ordered_docs_) {
      if (auto* ref = get_if<DocResult::DocReference>(&doc->value); ref) {
        incomplete_shards.insert(ref->shard_id);
        ref->requested = true;
      }
    }
    return incomplete_shards;
  }

 private:
  ConnectionContext* cntx_;
  std::string_view index_name_;
  search::SearchAlgorithm* search_algo_;
  SearchParams params_;

  size_t hops_ = 0;

  std::vector<absl::Duration> sharded_times_;
  std::vector<DocResult*> ordered_docs_;
  std::vector<SearchResult> sharded_results_;
};

}  // namespace

void SearchFamily::FtCreate(CmdArgList args, ConnectionContext* cntx) {
  DocIndex index{};

  CmdArgParser parser{args};
  string_view idx_name = parser.Next();

  while (parser.ToUpper().HasNext()) {
    // ON HASH | JSON
    if (parser.Check("ON").ExpectTail(1)) {
      index.type =
          parser.ToUpper().Next().Case("HASH"sv, DocIndex::HASH).Case("JSON"sv, DocIndex::JSON);
      continue;
    }

    // PREFIX count prefix [prefix ...]
    if (parser.Check("PREFIX").ExpectTail(2)) {
      if (size_t num = parser.Next().Int<size_t>(); num != 1)
        return (*cntx)->SendError("Multiple prefixes are not supported");
      index.prefix = string(parser.Next());
      continue;
    }

    // SCHEMA
    if (parser.Check("SCHEMA")) {
      auto schema = ParseSchemaOrReply(index.type, parser.Tail(), cntx);
      if (!schema)
        return;
      index.schema = move(*schema);
      break;  // SCHEMA always comes last
    }

    // Unsupported parameters are ignored for now
    parser.Skip(1);
  }

  if (auto err = parser.Error(); err)
    return (*cntx)->SendError(err->MakeReply());

  auto idx_ptr = make_shared<DocIndex>(move(index));
  cntx->transaction->ScheduleSingleHop([idx_name, idx_ptr](auto* tx, auto* es) {
    es->search_indices()->InitIndex(tx->GetOpArgs(es), idx_name, idx_ptr);
    return OpStatus::OK;
  });

  (*cntx)->SendOk();
}

void SearchFamily::FtDropIndex(CmdArgList args, ConnectionContext* cntx) {
  string_view idx_name = ArgS(args, 0);
  // TODO: Handle optional DD param

  atomic_uint num_deleted{0};
  cntx->transaction->ScheduleSingleHop([&](Transaction* t, EngineShard* es) {
    if (es->search_indices()->DropIndex(idx_name))
      num_deleted.fetch_add(1);
    return OpStatus::OK;
  });

  DCHECK(num_deleted == 0u || num_deleted == shard_set->size());
  if (num_deleted == shard_set->size())
    return (*cntx)->SendOk();
  (*cntx)->SendError("Unknown Index name");
}

void SearchFamily::FtInfo(CmdArgList args, ConnectionContext* cntx) {
  string_view idx_name = ArgS(args, 0);

  atomic_uint num_notfound{0};
  vector<DocIndexInfo> infos(shard_set->size());

  cntx->transaction->ScheduleSingleHop([&](Transaction* t, EngineShard* es) {
    auto* index = es->search_indices()->GetIndex(idx_name);
    if (index == nullptr)
      num_notfound.fetch_add(1);
    else
      infos[es->shard_id()] = index->GetInfo();
    return OpStatus::OK;
  });

  DCHECK(num_notfound == 0u || num_notfound == shard_set->size());

  if (num_notfound > 0u)
    return (*cntx)->SendError("Unknown index name");

  DCHECK(infos.front().base_index.schema.fields.size() ==
         infos.back().base_index.schema.fields.size());

  size_t total_num_docs = 0;
  for (const auto& info : infos)
    total_num_docs += info.num_docs;

  const auto& schema = infos.front().base_index.schema;

  (*cntx)->StartCollection(3, RedisReplyBuilder::MAP);

  (*cntx)->SendSimpleString("index_name");
  (*cntx)->SendSimpleString(idx_name);

  (*cntx)->SendSimpleString("fields");
  (*cntx)->StartArray(schema.fields.size());
  for (const auto& [field_ident, field_info] : schema.fields) {
    string_view reply[6] = {"identifier", string_view{field_ident},
                            "attribute",  field_info.short_name,
                            "type"sv,     SearchFieldTypeToString(field_info.type)};
    (*cntx)->SendSimpleStrArr(reply);
  }

  (*cntx)->SendSimpleString("num_docs");
  (*cntx)->SendLong(total_num_docs);
}

void SearchFamily::FtList(CmdArgList args, ConnectionContext* cntx) {
  atomic_int first{0};
  vector<string> names;

  cntx->transaction->ScheduleSingleHop([&](Transaction* t, EngineShard* es) {
    // Using `first` to assign `names` only once without a race
    if (first.fetch_add(1) == 0)
      names = es->search_indices()->GetIndexNames();
    return OpStatus::OK;
  });

  (*cntx)->SendStringArr(names);
}

void SearchFamily::FtSearch(CmdArgList args, ConnectionContext* cntx) {
  string_view index_name = ArgS(args, 0);
  string_view query_str = ArgS(args, 1);

  auto params = ParseSearchParamsOrReply(args.subspan(2), cntx);
  if (!params.has_value())
    return;

  search::SearchAlgorithm search_algo;
  search::SortOption* sort_opt = params->sort_option.has_value() ? &*params->sort_option : nullptr;
  if (!search_algo.Init(query_str, &params->query_params, sort_opt))
    return (*cntx)->SendError("Query syntax error");

  MultishardSearch{cntx, index_name, &search_algo, std::move(*params)}.RunAndReply();
}

void SearchFamily::FtProfile(CmdArgList args, ConnectionContext* cntx) {
  string_view index_name = ArgS(args, 0);
  string_view query_str = ArgS(args, 3);

  optional<SearchParams> params = ParseSearchParamsOrReply(args.subspan(4), cntx);
  if (!params.has_value())
    return;

  search::SearchAlgorithm search_algo;
  search::SortOption* sort_opt = params->sort_option.has_value() ? &*params->sort_option : nullptr;
  if (!search_algo.Init(query_str, &params->query_params, sort_opt))
    return (*cntx)->SendError("Query syntax error");

  search_algo.EnableProfiling();

  absl::Time start = absl::Now();

  CapturingReplyBuilder crb{facade::ReplyMode::ONLY_ERR};
  MultishardSearch mss{cntx, index_name, &search_algo, std::move(*params)};

  {
    CapturingReplyBuilder::ScopeCapture capture{&crb, cntx};
    mss.RunAndReply();
  }

  auto reply = crb.Take();
  if (auto err = CapturingReplyBuilder::GetError(reply); err)
    return (*cntx)->SendError(err->first, err->second);

  auto took = absl::Now() - start;

  auto profile = mss.GetProfileInfo();

  (*cntx)->StartArray(profile.profiles.size() + 1);

  // General stats
  (*cntx)->StartCollection(5, RedisReplyBuilder::MAP);
  (*cntx)->SendBulkString("took");
  (*cntx)->SendLong(absl::ToInt64Microseconds(took));
  (*cntx)->SendBulkString("hits");
  (*cntx)->SendLong(profile.total);
  (*cntx)->SendBulkString("serialized");
  (*cntx)->SendLong(profile.serialized);
  (*cntx)->SendSimpleString("cutoff");
  (*cntx)->SendLong(profile.cutoff);
  (*cntx)->SendSimpleString("hops");
  (*cntx)->SendLong(profile.hops);

  // Per-shard stats
  for (const auto& [profile, shard_took] : profile.profiles) {
    (*cntx)->StartCollection(2, RedisReplyBuilder::MAP);
    (*cntx)->SendBulkString("took");
    (*cntx)->SendLong(absl::ToInt64Microseconds(shard_took));
    (*cntx)->SendBulkString("tree");

    for (size_t i = 0; i < profile.events.size(); i++) {
      const auto& event = profile.events[i];

      size_t children = 0;
      for (size_t j = i + 1; j < profile.events.size(); j++) {
        if (profile.events[j].depth == event.depth)
          break;
        if (profile.events[j].depth == event.depth + 1)
          children++;
      }

      if (children > 0)
        (*cntx)->StartArray(2);

      (*cntx)->SendSimpleString(
          absl::StrFormat("t=%-10u n=%-10u %s", event.micros, event.num_processed, event.descr));

      if (children > 0)
        (*cntx)->StartArray(children);
    }
  }
}

#define HFUNC(x) SetHandler(&SearchFamily::x)

// Redis search is a module. Therefore we introduce dragonfly extension search
// to set as the default for the search family of commands. More sensible defaults,
// should also be considered in the future

void SearchFamily::Register(CommandRegistry* registry) {
  using CI = CommandId;

  // Disable journaling, because no-key-transactional enables it by default
  const uint32_t kReadOnlyMask =
      CO::NO_KEY_TRANSACTIONAL | CO::NO_KEY_TX_SPAN_ALL | CO::NO_AUTOJOURNAL;

  registry->StartFamily();
  *registry << CI{"FT.CREATE", CO::GLOBAL_TRANS, -2, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtCreate)
            << CI{"FT.DROPINDEX", CO::GLOBAL_TRANS, -2, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtDropIndex)
            << CI{"FT.INFO", kReadOnlyMask, 2, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtInfo)
            // Underscore same as in RediSearch because it's "temporary" (long time already)
            << CI{"FT._LIST", kReadOnlyMask, 1, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtList)
            << CI{"FT.SEARCH", kReadOnlyMask, -3, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtSearch)
            << CI{"FT.PROFILE", kReadOnlyMask, -4, 0, 0, 0, acl::FT_SEARCH}.HFUNC(FtProfile);
}

}  // namespace dfly
