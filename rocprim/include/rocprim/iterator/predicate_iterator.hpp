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
#include <type_traits>

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
/// \tparam PredicateDataIterator Type of the test iterator used to test the predicate function.
/// \tparam UnaryPredicate Type of the predicate function that tests the test.
template<class DataIterator, class PredicateDataIterator, class UnaryPredicate>
class predicate_iterator
{
public:
    static constexpr bool can_capture_reference
        = std::is_reference<decltype(*std::declval<DataIterator>())>::value;

    /// \brief A struct representing a reference that can be conditionally discarded.
    ///
    /// This struct holds a reference and a boolean flag.
    /// When assigning a value to the reference, it will only be assigned if the flag is set.
    /// When converted to the underlying value type, it will return the referenced value or the
    /// default-constructed value of the value type.
    /// If ``as_reference`` is ``false`` it instead uses a value instead of a reference.
    ///
    /// \tparam as_reference Boolean flag that decides whether the value is kept by reference or as value.
    template<bool as_reference = true>
    struct discard_underlying
    {
    public:
        /// \brief The type of the value that can be obtained by dereferencing the iterator.
        using value_type = typename std::iterator_traits<DataIterator>::value_type;

        /// \brief A reference type of the type iterated over (``value_type``).
        using reference = typename std::iterator_traits<DataIterator>::reference;

        /// \brief The internal type.
        using proxy = std::conditional_t<as_reference, reference, value_type>;

        /// \brief Constructs a ``discard_underlying`` object with the given reference and keep flag.
        /// \param ref The reference to be held.
        /// \param keep Boolean flag that indicates whether to keep the reference.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_underlying(proxy val, const bool keep)
            : underlying_(val), keep_(keep)
        {}

        /// \brief Assigns a value to the held reference if the keep flag is ``true``.
        /// \param value The value to assign to the reference.
        /// \return A reference to the (possibly) modified ``discard_underlying`` object.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE discard_underlying& operator=(const value_type& value)
        {
            if(keep_)
            {
                underlying_ = value;
            }
            return *this;
        }

        /// \brief Converts the ``discard_underlying`` to the underlying value type.
        /// \return The referenced value or the default-constructed value.
        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE operator value_type() const
        {
            return keep_ ? underlying_ : value_type{};
        }

    private:
        /// \brief The reference being held.
        proxy underlying_;

        /// \brief Boolean flag indicating wether to keep the reference or discard it.
        bool keep_;
    };

    using proxy = discard_underlying<can_capture_reference>;

    /// \brief The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename proxy::value_type;

    /// \brief A reference type of the type iterated over (``value_type``).
    using reference = typename proxy::reference;

    /// \brief A pointer type of the type iterated over (``value_type``).
    using pointer = typename std::iterator_traits<DataIterator>::pointer;

    /// \brief A type used for identify distance between iterators.
    using difference_type = typename std::iterator_traits<DataIterator>::difference_type;

    /// \brief The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    /// \brief The type of the test value that can be obtained by dereferencing the iterator.
    using predicate_data_value_type =
        typename std::iterator_traits<PredicateDataIterator>::value_type;

    /// \brief The type of predicate function used to select input range.
    using unary_predicate = UnaryPredicate;

    /// \brief Creates a new predicate_iterator.
    ///
    /// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
    /// \param predicate_iterator The test iterator that is used to test the predicate on.
    /// \param predicate Unary function used to select values obtained.
    /// from range pointed by \p iterator.
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator(DataIterator          data_iterator,
                                                          PredicateDataIterator predicate_iterator,
                                                          unary_predicate       predicate)
        : data_it_(data_iterator), predicate_data_it_(predicate_iterator), predicate_(predicate)
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
        predicate_data_it_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator++(int)
    {
        predicate_iterator old = *this;
        data_it_++;
        predicate_data_it_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator--()
    {
        data_it_--;
        predicate_data_it_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator--(int)
    {
        predicate_iterator old = *this;
        data_it_--;
        predicate_data_it_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE proxy operator*() const
    {
        return proxy(*data_it_, predicate_(*predicate_data_it_));
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE proxy operator->() const
    {
        return *(*this);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE proxy operator[](difference_type distance) const
    {
        return *(*this + distance);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator+(difference_type distance) const
    {
        return predicate_iterator(data_it_ + distance, predicate_data_it_ + distance, predicate_);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator+=(difference_type distance)
    {
        data_it_ += distance;
        predicate_data_it_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator operator-(difference_type distance) const
    {
        return predicate_iterator(data_it_ - distance, predicate_data_it_ - distance, predicate_);
    }

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE predicate_iterator& operator-=(difference_type distance)
    {
        data_it_ -= distance;
        predicate_data_it_ -= distance;
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
    DataIterator          data_it_;
    PredicateDataIterator predicate_data_it_;
    const UnaryPredicate  predicate_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class DataIterator, class PredicateDataIterator, class UnaryPredicate>
ROCPRIM_HOST_DEVICE inline predicate_iterator<DataIterator, PredicateDataIterator, UnaryPredicate>
    operator+(
        typename predicate_iterator<DataIterator, PredicateDataIterator, UnaryPredicate>::
            difference_type                                                            distance,
        const predicate_iterator<DataIterator, PredicateDataIterator, UnaryPredicate>& iterator)
{
    return iterator + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam PredicateDataIterator Type of ``predicate_data_iterator``.
/// \tparam UnaryPredicate Type of ``predicate``.
///
/// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
/// \param predicate_data_iterator The test iterator that is used to test the predicate on.
/// \param predicate The predicate function.
template<class DataIterator, class PredicateDataIterator, class UnaryPredicate>
auto make_predicate_iterator(DataIterator          data_iterator,
                             PredicateDataIterator predicate_data_iterator,
                             UnaryPredicate        predicate)
{
    return predicate_iterator<DataIterator, PredicateDataIterator, UnaryPredicate>(
        data_iterator,
        predicate_data_iterator,
        predicate);
}

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam UnaryPredicate Type of ``predicate``.
///
/// \param data_iterator The data iterator that will be forwarded whenever the predicate is true.
/// \param predicate The predicate function. It will be tested on ``data_iterator``.
template<class DataIterator, class UnaryPredicate>
ROCPRIM_HOST_DEVICE inline predicate_iterator<DataIterator, DataIterator, UnaryPredicate>
    make_predicate_iterator(DataIterator data_iterator, UnaryPredicate predicate)
{
    return make_predicate_iterator<DataIterator, DataIterator>(data_iterator,
                                                               data_iterator,
                                                               predicate);
}

/// \brief Constructs a ``predicate_iterator`` which can discard values assigned to it upon dereference based on a predicate.
///
/// \tparam DataIterator Type of ``data_iterator``.
/// \tparam FlagIterator Type of ``flag_iterator``. Its ``value_type`` should be implicitely be convertible to ``bool``.
///
/// \param data_iterator The data iterator that will be forwarded when the corresponding flag is set to ``true``.
/// \param flag_iterator The flag iterator.
template<class DataIterator, class FlagIterator>
auto make_mask_iterator(DataIterator data_iterator, FlagIterator flag_iterator)
{
    return make_predicate_iterator(data_iterator,
                                   flag_iterator,
                                   [] ROCPRIM_HOST_DEVICE(bool value) { return value; });
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_PREDICATE_ITERATOR_HPP_
