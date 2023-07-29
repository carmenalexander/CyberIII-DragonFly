// Copyright 2023, Roman Gershman.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include "base/pmr/memory_resource.h"
#include "core/detail/bptree_internal.h"

namespace dfly {

template <typename T> struct DefaultCompareTo {
  int operator()(const T& a, const T& b) const {
    std::less<T> cmp;
    return cmp(a, b) ? -1 : (cmp(b, a) ? 1 : 0);
  }
};

template <typename T> struct BPTreePolicy {
  using KeyT = T;
  using KeyCompareTo = DefaultCompareTo<T>;
};

template <typename T, typename Policy = BPTreePolicy<T>> class BPTree {
  BPTree(const BPTree&) = delete;
  BPTree& operator=(const BPTree&) = delete;

  using BPTreeNode = detail::BPTreeNode<Policy>;
  using BPTreePath = detail::BPTreePath<Policy>;

 public:
  using KeyT = typename Policy::KeyT;

  BPTree(PMR_NS::memory_resource* mr = PMR_NS::get_default_resource()) : mr_(mr) {
  }

  ~BPTree() {
    Clear();
  }

  // true if inserted, false if skipped.
  bool Insert(KeyT item);

  bool Contains(KeyT item) const;

  size_t Height() const {
    return height_;
  }

  size_t Size() const {
    return count_;  // number of items in the tree
  }

  size_t NodeCount() const {
    // number of nodes in the tree (usually, order of magnitude smaller than Size()).
    return num_nodes_;
  }

  void Clear();

  BPTreeNode* DEBUG_root() {
    return root_;
  }

 private:
  BPTreeNode* CreateNode(bool leaf);

  void DestroyNode(BPTreeNode* node);

  // Unloads the full leaf to allow insertion of additional item.
  // The leaf should be the last one in the path.
  std::pair<BPTreeNode*, KeyT> InsertFullLeaf(KeyT item, const BPTreePath& path);

  // Charts the path towards key. Returns true if key is found.
  // In that case path->Last().first->Key(path->Last().second) == key.
  // Fills the tree path not including the key itself.
  bool Locate(KeyT key, BPTreePath* path) const;

  BPTreeNode* root_ = nullptr;  // root node or NULL if empty tree
  uint32_t count_ = 0;          // number of items in tree
  uint32_t height_ = 0;         // height of tree from root to leaf
  uint32_t num_nodes_ = 0;      // number of nodes in tree
  PMR_NS::memory_resource* mr_;
};

template <typename T, typename Policy> bool BPTree<T, Policy>::Contains(KeyT item) const {
  BPTreePath path;
  bool found = Locate(item, &path);
  return found;
}

template <typename T, typename Policy> void BPTree<T, Policy>::Clear() {
  if (!root_)
    return;

  BPTreePath path;
  BPTreeNode* node = root_;

  auto deep_left = [&](unsigned pos) {
    do {
      path.Push(node, pos);
      node = node->Child(pos);
      pos = 0;
    } while (!node->IsLeaf());
  };

  if (!root_->IsLeaf())
    deep_left(0);

  while (true) {
    DestroyNode(node);

    if (path.Depth() == 0) {
      break;
    }
    node = path.Last().first;
    unsigned pos = path.Last().second;
    path.Pop();
    if (pos < node->NumItems()) {
      deep_left(pos + 1);
    }
  }
  root_ = nullptr;
  height_ = count_ = 0;
}

template <typename T, typename Policy> bool BPTree<T, Policy>::Insert(KeyT item) {
  if (!root_) {
    root_ = CreateNode(true);
    root_->InitSingle(item);
    count_ = height_ = 1;

    return true;
  }

  BPTreePath path;
  bool found = Locate(item, &path);

  if (found) {
    return false;
  }

  assert(path.Depth() > 0u);

  BPTreeNode* leaf = path.Last().first;
  assert(leaf->IsLeaf());

  if (leaf->NumItems() == BPTreeNode::kMaxLeafKeys) {
    unsigned root_free [[maybe_unused]] = root_->AvailableSlotCount();
    std::pair<BPTreeNode*, KeyT> res = InsertFullLeaf(item, path);
    if (res.first) {  // we propagated the new node all the way to the root.
      assert(root_free == 0u);
      BPTreeNode* new_root = CreateNode(false);
      new_root->InitSingle(res.second);
      new_root->SetChild(0, root_);
      new_root->SetChild(1, res.first);
      root_ = new_root;
      height_++;
    }
  } else {
    unsigned pos = path.Last().second;
    leaf->LeafInsert(pos, item);
  }
  count_++;
  return true;
}

template <typename T, typename Policy>
bool BPTree<T, Policy>::Locate(KeyT key, BPTreePath* path) const {
  assert(root_);
  BPTreeNode* node = root_;
  while (true) {
    typename BPTreeNode::SearchResult res = node->BSearch(key);
    path->Push(node, res.index);
    if (res.found) {
      return true;
    }
    assert(res.index <= node->NumItems());

    if (node->IsLeaf()) {
      break;
    }
    node = node->Child(res.index);
  }
  return false;
}

template <typename T, typename Policy>
auto BPTree<T, Policy>::InsertFullLeaf(KeyT item, const BPTreePath& path)
    -> std::pair<BPTreeNode*, KeyT> {
  assert(path.Depth() > 0u);

  BPTreeNode* node = path.Last().first;
  assert(node->IsLeaf() && node->AvailableSlotCount() == 0);

  unsigned insert_pos = path.Last().second;
  unsigned level = path.Depth() - 1;
  if (level > 0) {
    BPTreeNode* parent = path.Node(level - 1);
    unsigned pos = path.Position(level - 1);
    assert(parent->Child(pos) == node);

    std::pair<BPTreeNode*, unsigned> rebalance_res = parent->RebalanceChild(pos, insert_pos);
    if (rebalance_res.first) {
      rebalance_res.first->LeafInsert(rebalance_res.second, item);
      return {nullptr, 0};
    }
  }

  KeyT median;
  BPTreeNode* right = CreateNode(node->IsLeaf());
  node->Split(right, &median);

  assert(node->NumItems() < BPTreeNode::kMaxLeafKeys);

  if (insert_pos <= node->NumItems()) {
    assert(item < median);
    node->LeafInsert(insert_pos, item);
  } else {
    assert(item > median);
    right->LeafInsert(insert_pos - node->NumItems() - 1, item);
  }

  // we now must add right to the paren if it exists.
  while (level-- > 0) {
    node = path.Node(level);            // level up, now node is parent.
    insert_pos = path.Position(level);  // insert_pos is position of node in parent.

    assert(!node->IsLeaf() && insert_pos <= node->NumItems());

    if (node->NumItems() == BPTreeNode::kMaxInnerKeys) {
      if (level > 0) {
        BPTreeNode* parent = path.Node(level - 1);
        unsigned node_pos = path.Position(level - 1);
        assert(parent->Child(node_pos) == node);
        std::pair<BPTreeNode*, unsigned> rebalance_res =
            parent->RebalanceChild(node_pos, insert_pos);
        if (rebalance_res.first) {
          rebalance_res.first->InnerInsert(rebalance_res.second, median, right);
          return {nullptr, 0};
        }
      }

      KeyT parent_median;
      BPTreeNode* parent_right = CreateNode(false);
      node->Split(parent_right, &parent_median);
      assert(node->NumItems() < BPTreeNode::kMaxInnerKeys);

      if (insert_pos <= node->NumItems()) {
        assert(median < parent_median);
        node->InnerInsert(insert_pos, median, right);
      } else {
        assert(median > parent_median);
        parent_right->InnerInsert(insert_pos - node->NumItems() - 1, median, right);
      }
      right = parent_right;
      median = parent_median;
    } else {
      node->InnerInsert(insert_pos, median, right);
      return {nullptr, 0};
    }
  }

  return {right, median};
}

template <typename T, typename Policy>
detail::BPTreeNode<Policy>* BPTree<T, Policy>::CreateNode(bool leaf) {
  num_nodes_++;
  void* ptr = mr_->allocate(BPTreeNode::kTargetNodeSize, 8);
  BPTreeNode* node = new (ptr) BPTreeNode(leaf);

  return node;
}

template <typename T, typename Policy> void BPTree<T, Policy>::DestroyNode(BPTreeNode* node) {
  void* ptr = node;
  mr_->deallocate(ptr, BPTreeNode::kTargetNodeSize, 8);
  num_nodes_--;
}

}  // namespace dfly
