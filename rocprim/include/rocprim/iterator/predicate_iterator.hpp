// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_ITERATOR_PREDICATE_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_PREDICATE_ITERATOR_HPP_

#include "../config.hpp"

#include <iterator>

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class predicate_iterator
/// \brief A random-access iterator which can discard values assigned to it upon dereference based on a predicate.
///
/// \par Overview
/// * ``predicate_iterator`` can be used to ignore certain input or output of algorithms.
/// * When writing to ``predicate_iterator``, it will only write to the underlying iterator if the predicate holds.
///   Otherwise it will discard the value.
/// * When reading from ``predicate_iterator``, it will only read from the underlying iterator if the predicate holds.
///   Otherwise it will return the default constructed value.
///
/// \tparam DataIterator Type of the data iterator that will be forwarded upon dereference.
/// \tparam TestIterator Type of the test iterator used to test the predicate function.
/// \tparam PredicateFunction Type of the predicate function that tests the test.
template<class DataIterator, class TestIterator, class PredicateFunction>
class predicate_iterator
{
public:
    /// \brief The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::iterator_traits<DataIterator>::value_type;

    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `const` since predicate_iterator is a read-only iterator.
    using reference = typename std::iterator_traits<DataIterator>::reference;

    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since predicate_iterator is a read-only iterator.
    using pointer = typename std::iterator_traits<DataIterator>::pointer;

    /// \brief A type used for identify distance between iterators.
    using difference_type = typename std::iterator_traits<DataIterator>::difference_type;

    /// \brief The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    /// \brief The type of the test value that can be obtained by dereferencing the iterator.
    using test_type = typename std::iterator_traits<TestIterator>::value_type;

    /// \brief The type of predicate function used to select input range.
    using predicate_function = PredicateFunction;

    /// \brief A struct representing a reference that can be conditionally discarded.
    ///
    /// This struct holds a reference and a boolean flag.
    /// When assigning a value to the reference, it will only be assigned if the flag is set.
    /// When converted to the underlying value type, it will return the referenced value or the
    /// default-constructed value of the value type.
    struct discard_reference
    {
    private:
        /// \brief The reference being held.
        reference ref_;

        /// \brief Boolean flag indicating wether to keep the reference or discard it.
        const bool keep_;

    public:
        /// \brief Constructs a ``discard_reference`` object with the given reference and keep flag.
        /// \param ref The reference to be held.
        /// \param keep Boolean flag that indicates whether to keep the reference.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_reference(reference ref, const bool keep)
            : ref_(ref), keep_(keep)
        {}

        /// \brief Assigns a value to the held reference if the keep flag is ``true``.
        /// \param value The value to assign to the reference.
        /// \return A reference to the (possibly) modified ``discard_reference`` object.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_reference& operator=(const value_type& value)
        {
            if(keep_)
            {
                ref_ = value;
            }
            return *this;
        }

        /// \brief Converts the ``discard_reference`` to the underlying value type.
        /// \return The referenced value or the default-constructed value.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE operator value_type() const
        {
            return keep_ ? ref_ : value_type{};
        }
    };

    /// \brief Creates a new predicate_iterator.
    ///
    /// \param iterator input iterator to iterate over and transform.
    /// \param predicate unary function used to select values obtained
    /// from range pointed by \p iterator.
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
        predicate_iterator(DataIterator iterator, TestIterator test, predicate_function predicate)
        : data_it_(iterator), test_it_(test), predicate_(predicate)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    // Default rule of three, as MSVC thinks the copy assignment operator is implicitly deleted.
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE ~predicate_iterator()                         = default;
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator(const predicate_iterator&) = default;
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator=(const predicate_iterator&)
        = default;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator++()
    {
        data_it_++;
        test_it_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator++(int)
    {
        predicate_iterator old = *this;
        data_it_++;
        test_it_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator--()
    {
        data_it_--;
        test_it_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator--(int)
    {
        predicate_iterator old = *this;
        data_it_--;
        test_it_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_reference operator*() const
    {
        return discard_reference(*data_it_, predicate_(*test_it_));
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_reference operator->() const
    {
        return *(*this);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_reference operator[](difference_type distance) const
    {
        return *(*this + distance);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator+(difference_type distance) const
    {
        return predicate_iterator(data_it_ + distance, test_it_ + distance, predicate_);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator+=(difference_type distance)
    {
        data_it_ += distance;
        test_it_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator-(difference_type distance) const
    {
        return predicate_iterator(data_it_ - distance, test_it_ - distance, predicate_);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator-=(difference_type distance)
    {
        data_it_ -= distance;
        test_it_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE difference_type operator-(predicate_iterator other) const
    {
        return data_it_ - other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator==(predicate_iterator other) const
    {
        return data_it_ == other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator!=(predicate_iterator other) const
    {
        return data_it_ != other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator<(predicate_iterator other) const
    {
        return data_it_ < other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator<=(predicate_iterator other) const
    {
        return data_it_ <= other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator>(predicate_iterator other) const
    {
        return data_it_ > other.data_it_;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE bool operator>=(predicate_iterator other) const
    {
        return data_it_ >= other.data_it_;
    }

    friend std::ostream& operator<<(std::ostream& os, const predicate_iterator& /* iter */)
    {
        return os;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    DataIterator            data_it_;
    TestIterator            test_it_;
    const PredicateFunction predicate_;
};

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam TestIterator Type of ``test_iterator``.
/// \tparam PredicateFunction Type of ``predicate``.
///
/// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
/// \param test_iterator The test iterator that is used to test the predicate on.
/// \param predicate The predicate function.
template<class DataIterator, class TestIterator, class PredicateFunction>
auto make_predicate_iterator(DataIterator      data_iterator,
                             TestIterator      test_iterator,
                             PredicateFunction predicate)
{
    return predicate_iterator<DataIterator, TestIterator, PredicateFunction>(data_iterator,
                                                                             test_iterator,
                                                                             predicate);
}

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam PredicateFunction Type of ``predicate``.
///
/// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
/// \param predicate The predicate function. It will be tested on ``data_iterator``.
template<class DataIterator, class PredicateFunction>
ROCPRIM_HOST_DEVICE inline predicate_iterator<DataIterator, DataIterator, PredicateFunction>
    make_predicate_iterator(DataIterator data_iterator, PredicateFunction predicate)
{
    return make_predicate_iterator<DataIterator, DataIterator>(data_iterator,
                                                               data_iterator,
                                                               predicate);
}

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam BoolIterator Type of ``mask_iterator``. Should iterate over boolean values.
///
/// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
/// \param mask_iterator The test iterator that is used to test the predicate on.
template<class DataIterator, class BoolIterator>
auto make_mask_iterator(DataIterator data_iterator, BoolIterator mask_iterator)
{
    return make_predicate_iterator(data_iterator,
                                   mask_iterator,
                                   [] ROCPRIM_HOST_DEVICE(bool value) { return value; });
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_PREDICATE_ITERATOR_HPP_
