// Copyright 2023, Roman Gershman.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include <array>
#include <cassert>
#include <cstring>

namespace dfly {

template <typename T, typename Policy> class BPTree;

namespace detail {

// Internal classes related to B+tree implementation. The design is largely based on the
// implementation of absl::bPtree_map/set.
// The motivation for replacing zskiplist - significant size reduction:
//   we reduce the metadata overhead per record from 45 bytes in zskiplist to just a
//   few bytes with b-tree. The trick is using significantly large nodes (256 bytes) so that
//   their overhead is negligible compared to the items they store.
//   Why not use absl::bPtree_set? We must support Rank tree functionality that
//   absl does not supply.
//   Hacking into absl is not a simple task, implementing our own tree is easier.
// Below some design decisions:
// 1. We use predefined node size of 256 bytes and derive number of items in each node from it.
//    Inner nodes have less items than leaf nodes because they also need to store child pointers.
// 2. BPTreeNode does not predeclare fields besides the 8 bytes metadata - everything else is
//    calculated at run-time and has dedicated accessors (similarly to absl). This allows
//    dense and efficient representation of tree nodes.
// 3. We assume that we store small items (8, 16 bytes) which will have a large branching
//    factor (248/16), meaning the tree will stay shallow even for sizes reaching billion nodes.
// 4. We do not store parent pointer like in absl tree. Instead we use BPTreePath to store
//    hierarchy of parent nodes. That should reduce our overhead even further by few bits per item.
// 5. We assume we store trivially copyable types - this reduces the
//    complexity of the generics in the code.
// 6. We support pmr memory resource. This allows us to use pluggable heaps.
//
// TODO: (all the ideas taken from absl implementation)
//       1. to introduce slices when removing items from the tree (avoid shifts).
//       2. to avoid merging/rebalancing when removing max/min items from the tree.
//       3. Small tree optimization: when the tree is small with a single root node, we can
//          allocate less then 256 bytes (special case) to avoid relative blowups in memory for
//          small trees.

template <typename Policy> class BPTreeNode {
  using _Key = typename Policy::KeyT;

  static_assert(std::is_trivially_copyable<_Key>::value, "KeyT must be triviall copyable");

  friend class ::dfly::BPTree<_Key, Policy>;

  BPTreeNode(const BPTreeNode&) = delete;
  BPTreeNode& operator=(const BPTreeNode&) = delete;

  BPTreeNode(bool leaf) : num_items_(0), leaf_(leaf) {
  }

  static constexpr uint16_t kKeySize = sizeof(_Key);
  static constexpr uint16_t kTargetNodeSize = 256;
  static constexpr uint16_t kKeyOffset = sizeof(uint64_t);  // 8 bytes for metadata

 public:
  static constexpr uint16_t kMaxLeafKeys = (kTargetNodeSize - kKeyOffset) / kKeySize;
  static constexpr uint16_t kMinLeafKeys = kMaxLeafKeys / 2;

  // internal node:
  // x slots, (x+1) children: x * kKeySize + (x+1) * sizeof(BPTreeNode*) = x * (kKeySize + 8) + 8
  // x = (kTargetNodeSize - 8 - kKeyOffset) / (kKeySize + 8)

  static constexpr uint16_t kMaxInnerKeys =
      (kTargetNodeSize - sizeof(void*) - kKeyOffset) / (kKeySize + sizeof(void*));
  static constexpr uint16_t kMinInnerKeys = kMaxInnerKeys / 2;

  using KeyT = _Key;

  void InitSingle(KeyT key) {
    SetKey(0, key);
    num_items_ = 1;
  }

  KeyT Key(unsigned index) const {
    KeyT res;
    memcpy(&res, KeyPtr(index), kKeySize);
    return res;
  }

  void SetKey(size_t index, KeyT item) {
    uint8_t* slot = KeyPtr(index);
    memcpy(slot, &item, kKeySize);
  }

  BPTreeNode** Children() {
    // aligned to 8 bytes.
    constexpr uint16_t kChildOffset = kMaxInnerKeys * kKeySize + kKeyOffset;

    uint8_t* ptr = reinterpret_cast<uint8_t*>(this) + kChildOffset;
    return reinterpret_cast<BPTreeNode**>(ptr);
  }

  BPTreeNode* Child(unsigned i) {
    return Children()[i];
  }

  void SetChild(unsigned i, BPTreeNode* child) {
    Children()[i] = child;
  }

  struct SearchResult {
    uint16_t index;
    bool found;
  };

  // Searches for key in the node using binary search.
  // Returns SearchResult with index of the key if found.
  SearchResult BSearch(KeyT key) const;

  void Split(BPTreeNode* right, KeyT* median);

  bool IsLeaf() const {
    return leaf_;
  }

  unsigned NumItems() const {
    return num_items_;
  }

  unsigned AvailableSlotCount() const {
    return MaxItems() - num_items_;
  }

  unsigned MaxItems() const {
    return IsLeaf() ? kMaxLeafKeys : kMaxInnerKeys;
  }

  unsigned MinItems() const {
    return IsLeaf() ? kMinLeafKeys : kMinInnerKeys;
  }

  void ShiftRight(unsigned index);

  // Rebalance a full child at position pos, at which we tried to insert at insert_pos.
  // Returns the node and the position to insert into if rebalancing succeeded.
  // Returns nullptr if rebalancing did not succeed.
  std::pair<BPTreeNode*, unsigned> RebalanceChild(unsigned pos, unsigned insert_pos);

  // Inserts item into a leaf node.
  // Assumes: the node is IsLeaf() and has some space.
  void LeafInsert(unsigned index, KeyT item) {
    assert(IsLeaf() && NumItems() < MaxItems());
    InsertItem(index, item);
  }

  void InnerInsert(unsigned index, KeyT item, BPTreeNode* child) {
    InsertItem(index, item);
    SetChild(index + 1, child);
  }

  // Tries to merge the child at position pos with its sibling.
  // If we did not succeed to merge, we try to rebalance.
  // Returns retired BPTreeNode* if children got merged and this parent node's children
  // count decreased, otherwise, we return nullptr (rebalanced).
  BPTreeNode* MergeOrRebalanceChild(unsigned pos);

  void Validate(KeyT upper_bound) const;

 private:
  void RebalanceChildToLeft(unsigned child_pos, unsigned count);
  void RebalanceChildToRight(unsigned child_pos, unsigned count);

  void MergeFromRight(KeyT key, BPTreeNode* right);

  void InsertItem(unsigned index, KeyT item) {
    assert(index <= num_items_);
    assert(index == 0 || Key(index - 1) < item);
    assert(index == num_items_ || Key(index) > item);

    ShiftRight(index);
    SetKey(index, item);
  }

  // The class is constructed inside a block of memory of size kTargetNodeSize.
  // Only BPTree can create it, hence it can access the memory outside its fields.
  uint8_t* KeyPtr(unsigned index) {
    return reinterpret_cast<uint8_t*>(this) + kKeyOffset + kKeySize * index;
  }

  const uint8_t* KeyPtr(unsigned index) const {
    return reinterpret_cast<const uint8_t*>(this) + kKeyOffset + kKeySize * index;
  }

  struct {
    uint64_t num_items_ : 7;
    uint64_t leaf_ : 1;
    uint64_t : 56;
  };

  static_assert(kMaxLeafKeys < 128, "num_items_ is 7 bits");
};

// Contains parent/index pairs. Meaning that node0->Child(index0) == node1.
template <typename Policy> class BPTreePath {
  static constexpr unsigned kMaxDepth = 16;

 public:
  void Push(BPTreeNode<Policy>* node, unsigned pos) {
    assert(depth_ < kMaxDepth);
    record_[depth_].node = node;
    record_[depth_].pos = pos;
    depth_++;
  }

  unsigned Depth() const {
    return depth_;
  }

  std::pair<BPTreeNode<Policy>*, unsigned> Last() const {
    assert(depth_ > 0u);
    return {record_[depth_ - 1].node, record_[depth_ - 1].pos};
  }

  BPTreeNode<Policy>* Node(unsigned i) const {
    assert(i < depth_);
    return record_[i].node;
  }

  unsigned Position(unsigned i) const {
    assert(i < depth_);
    return record_[i].pos;
  }

  void Pop() {
    assert(depth_ > 0u);
    depth_--;
  }

 private:
  struct Record {
    BPTreeNode<Policy>* node;
    unsigned pos;
  };

  std::array<Record, kMaxDepth> record_;
  unsigned depth_ = 0;
};

// Returns the position of the first item whose key is greater or equal than key.
// if all items are smaller than key, returns num_items_.
template <typename Policy> auto BPTreeNode<Policy>::BSearch(KeyT key) const -> SearchResult {
  uint16_t lo = 0;
  uint16_t hi = num_items_;
  typename Policy::KeyCompareTo cmp_op;
  while (lo < hi) {
    uint16_t mid = (lo + hi) >> 1;
    assert(mid < hi);

    KeyT item = Key(mid);

    int cmp_res = cmp_op(key, item);
    if (cmp_res == 0) {
      return SearchResult{.index = mid, .found = true};
    }

    if (cmp_res < 0) {
      hi = mid;
    } else {
      lo = mid + 1;  // we never return indices upto mid because they are strictly less than key.
    }
  }
  assert(lo == hi);

  return {.index = hi, .found = 0};
}

template <typename Policy> void BPTreeNode<Policy>::ShiftRight(unsigned index) {
  unsigned num_items_to_shift = num_items_ - index;
  if (num_items_to_shift > 0) {
    uint8_t* ptr = KeyPtr(index);
    memmove(ptr + kKeySize, ptr, num_items_to_shift * kKeySize);
    BPTreeNode** children = Children();
    if (!IsLeaf()) {
      memmove(children + index + 1, children + index,
              (num_items_to_shift + 1) * sizeof(BPTreeNode*));
    }
  }
  num_items_++;
}

/***
 *  Rebalances the (full) child at position pos with its sibling. `this` node is an inner node.
 *  It first tried to rebalance (move items) from the full child to its left sibling. If the left
 *  sibling does not have enough space, it tries to rebalance to the right sibling. The caller
 *  passes the original position of the item it tried to insert into the full child. In case the
 *  rebalance succeeds the function returns the new node and the position to insert into. Otherwise,
 *  it returns result.first == nullptr.
 */
template <typename Policy>
std::pair<BPTreeNode<Policy>*, unsigned> BPTreeNode<Policy>::RebalanceChild(unsigned pos,
                                                                            unsigned insert_pos) {
  unsigned to_move = 0;
  BPTreeNode* node = Child(pos);

  if (pos > 0) {
    BPTreeNode* left = Child(pos - 1);
    unsigned dest_free = left->AvailableSlotCount();
    if (dest_free > 0) {
      // We bias rebalancing based on the position being inserted. If we're
      // inserting at the end of the right node then we bias rebalancing to
      // fill up the left node.
      if (insert_pos == node->NumItems()) {
        to_move = dest_free;
        assert(to_move < node->NumItems());
      } else if (dest_free > 1) {
        // we move less than left free capacity which leaves as some space in the node.
        to_move = dest_free / 2;
      }

      if (to_move) {
        unsigned dest_old_count = left->NumItems();
        RebalanceChildToLeft(pos, to_move);
        assert(node->AvailableSlotCount() == to_move);
        if (insert_pos < to_move) {
          assert(left->AvailableSlotCount() > 0u);       // we did not fill up the left node.
          insert_pos = dest_old_count + insert_pos + 1;  // +1 because we moved the separator.
          node = left;
        } else {
          insert_pos -= to_move;
        }

        return {node, insert_pos};
      }
    }
  }

  if (pos < NumItems()) {
    BPTreeNode* right = Child(pos + 1);
    unsigned dest_free = right->AvailableSlotCount();
    if (dest_free > 0) {
      if (insert_pos == 0) {
        to_move = dest_free;
        assert(to_move < node->NumItems());
      } else if (dest_free > 1) {
        to_move = dest_free / 2;
      }

      if (to_move) {
        RebalanceChildToRight(pos, to_move);
        if (insert_pos > node->NumItems()) {
          insert_pos -= (node->NumItems() + 1);
          node = right;
        }
        return {node, insert_pos};
      }
    }
  }
  return {nullptr, 0};
}

template <typename Policy>
void BPTreeNode<Policy>::RebalanceChildToLeft(unsigned child_pos, unsigned count) {
  assert(child_pos > 0u);
  BPTreeNode* src = Child(child_pos);
  BPTreeNode* dest = Child(child_pos - 1);
  assert(src->NumItems() >= count);
  assert(count >= 1u);
  assert(dest->AvailableSlotCount() >= count);

  unsigned dest_items = dest->NumItems();

  // Move the delimiting value to the left node.
  dest->SetKey(dest_items, Key(child_pos - 1));

  // Copy src keys [0, count-1] to dest keys [dest_items+1, dest_items+count].
  for (unsigned i = 1; i < count; ++i) {
    dest->SetKey(dest_items + i, src->Key(i - 1));
  }

  SetKey(child_pos - 1, src->Key(count - 1));

  // Shift the values in the right node to their correct position.
  for (unsigned i = count; i < src->NumItems(); ++i) {
    src->SetKey(i - count, src->Key(i));
  }

  if (!src->IsLeaf()) {
    // Move the child pointers from the right to the left node.
    for (unsigned i = 0; i < count; ++i) {
      dest->SetChild(1 + dest->NumItems() + i, src->Child(i));
    }
    for (unsigned i = count; i <= src->NumItems(); ++i) {
      src->SetChild(i - count, src->Child(i));
      src->SetChild(i, NULL);
    }
  }

  // Fixup the counts on the src and dest nodes.
  dest->num_items_ += count;
  src->num_items_ -= count;
}

template <typename Policy>
void BPTreeNode<Policy>::RebalanceChildToRight(unsigned child_pos, unsigned count) {
  assert(child_pos < NumItems());
  BPTreeNode* src = Child(child_pos);
  BPTreeNode* dest = Child(child_pos + 1);

  assert(src->NumItems() >= count);
  assert(count >= 1u);
  assert(dest->AvailableSlotCount() >= count);

  unsigned dest_items = dest->NumItems();

  assert(dest_items > 0u);

  // Shift the values in the right node to their correct position.
  for (int i = dest_items - 1; i >= 0; --i) {
    dest->SetKey(i + count, dest->Key(i));
  }

  // Move the delimiting value to the left node and the new delimiting value
  // from the right node.
  KeyT new_delim = src->Key(src->NumItems() - count);
  for (unsigned i = 1; i < count; ++i) {
    unsigned src_id = src->NumItems() - count + i;
    dest->SetKey(i - 1, src->Key(src_id));
  }
  // Move parent's delimiter to destination and update it with new delimiter.
  dest->SetKey(count - 1, Key(child_pos));
  SetKey(child_pos, new_delim);

  if (!src->IsLeaf()) {
    // Shift child pointers in the right node to their correct position.
    for (int i = dest_items; i >= 0; --i) {
      dest->SetChild(i + count, dest->Child(i));
    }

    // Move child pointers from the left node to the right.
    for (unsigned i = 0; i < count; ++i) {
      unsigned src_id = src->NumItems() - (count - 1) + i;
      dest->SetChild(i, src->Child(src_id));
      src->SetChild(src_id, NULL);
    }
  }

  // Fixup the counts on the src and dest nodes.
  dest->num_items_ += count;
  src->num_items_ -= count;
}

// splits the node into two nodes. The left node is the current node and the right node is
// is filled with the right half of the items. The median key is returned in *median.
template <typename Policy> void BPTreeNode<Policy>::Split(BPTreeNode<Policy>* right, KeyT* median) {
  unsigned mid = num_items_ / 2;
  *median = Key(mid);
  right->leaf_ = leaf_;
  right->num_items_ = num_items_ - (mid + 1);
  memmove(right->KeyPtr(0), KeyPtr(mid + 1), right->num_items_ * kKeySize);
  if (!IsLeaf()) {
    BPTreeNode** rchild = right->Children();
    for (size_t i = 0; i <= right->num_items_; i++) {
      rchild[i] = Child(mid + 1 + i);
    }
  }
  num_items_ = mid;
}

}  // namespace detail
}  // namespace dfly
