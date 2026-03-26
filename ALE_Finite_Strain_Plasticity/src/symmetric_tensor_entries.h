#pragma once

#include <deal.II/base/symmetric_tensor.h>

#include <cstddef>
#include <iterator>

namespace dealii_utils
{
  // ---------------- Const view ----------------
  template <int rank, int dim, typename Number>
  class SymmetricTensorConstEntriesView
  {
  public:
    using tensor_type = dealii::SymmetricTensor<rank, dim, Number>;

    explicit SymmetricTensorConstEntriesView(const tensor_type &t) : t_(&t) {}

    class const_iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = Number;
      using difference_type   = std::ptrdiff_t;
      using pointer           = const Number *;
      using reference         = const Number &;

      const_iterator() = default;
      const_iterator(const tensor_type *t, unsigned int i) : t_(t), i_(i) {}

      reference operator*() const { return t_->access_raw_entry(i_); }

      const_iterator &operator++() { ++i_; return *this; }
      const_iterator operator++(int) { const_iterator tmp(*this); ++(*this); return tmp; }

      friend bool operator==(const const_iterator &a, const const_iterator &b)
      {
        return a.t_ == b.t_ && a.i_ == b.i_;
      }
      friend bool operator!=(const const_iterator &a, const const_iterator &b) { return !(a == b); }

    private:
      const tensor_type *t_ = nullptr;
      unsigned int       i_ = 0;
    };

    const_iterator begin() const { return const_iterator(t_, 0u); }
    const_iterator end() const { return const_iterator(t_, tensor_type::n_independent_components); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

  private:
    const tensor_type *t_;
  };


  // ---------------- Mutable view ----------------
  template <int rank, int dim, typename Number>
  class SymmetricTensorMutableEntriesView
  {
  public:
    using tensor_type = dealii::SymmetricTensor<rank, dim, Number>;

    explicit SymmetricTensorMutableEntriesView(tensor_type &t) : t_(&t) {}

    class iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = Number;
      using difference_type   = std::ptrdiff_t;
      using pointer           = Number *;
      using reference         = Number &;

      iterator() = default;
      iterator(tensor_type *t, unsigned int i) : t_(t), i_(i) {}

      reference operator*() const { return t_->access_raw_entry(i_); }

      iterator &operator++() { ++i_; return *this; }
      iterator operator++(int) { iterator tmp(*this); ++(*this); return tmp; }

      friend bool operator==(const iterator &a, const iterator &b)
      {
        return a.t_ == b.t_ && a.i_ == b.i_;
      }
      friend bool operator!=(const iterator &a, const iterator &b) { return !(a == b); }

    private:
      tensor_type   *t_ = nullptr;
      unsigned int   i_ = 0;
    };

    class const_iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = Number;
      using difference_type   = std::ptrdiff_t;
      using pointer           = const Number *;
      using reference         = const Number &;

      const_iterator() = default;
      const_iterator(const tensor_type *t, unsigned int i) : t_(t), i_(i) {}

      reference operator*() const { return t_->access_raw_entry(i_); }

      const_iterator &operator++() { ++i_; return *this; }
      const_iterator operator++(int) { const_iterator tmp(*this); ++(*this); return tmp; }

      friend bool operator==(const const_iterator &a, const const_iterator &b)
      {
        return a.t_ == b.t_ && a.i_ == b.i_;
      }
      friend bool operator!=(const const_iterator &a, const const_iterator &b) { return !(a == b); }

    private:
      const tensor_type *t_ = nullptr;
      unsigned int       i_ = 0;
    };

    iterator begin() { return iterator(t_, 0u); }
    iterator end() { return iterator(t_, tensor_type::n_independent_components); }

    const_iterator begin() const { return const_iterator(t_, 0u); }
    const_iterator end() const { return const_iterator(t_, tensor_type::n_independent_components); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

  private:
    tensor_type *t_;
  };


  // ---------------- Factories ----------------
  template <int rank, int dim, typename Number>
  SymmetricTensorConstEntriesView<rank, dim, Number>
  symmetric_tensor_entries(const dealii::SymmetricTensor<rank, dim, Number> &t)
  {
    return SymmetricTensorConstEntriesView<rank, dim, Number>(t);
  }

  template <int rank, int dim, typename Number>
  SymmetricTensorMutableEntriesView<rank, dim, Number>
  symmetric_tensor_entries(dealii::SymmetricTensor<rank, dim, Number> &t)
  {
    return SymmetricTensorMutableEntriesView<rank, dim, Number>(t);
  }
} // namespace dealii_utils
