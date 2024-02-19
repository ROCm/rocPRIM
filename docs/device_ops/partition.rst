.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-partition:

********************************************************************
 Partition
********************************************************************

partition
============

.. doxygenfunction:: rocprim::partition(void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, SelectedCountOutputIterator selected_count_output, const size_t size, UnaryPredicate predicate, const hipStream_t stream=0, const bool debug_synchronous=false)

partition_two_way
~~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::partition_two_way(void* temporary_storage, size_t& storage_size, InputIterator input, SelectedOutputIterator output_selected, RejectedOutputIterator output_rejected, SelectedCountOutputIterator selected_count_output, const size_t size, Predicate predicate, const hipStream_t stream = 0, const bool debug_synchronous = false)

partition_three_way
======================

.. doxygenfunction:: rocprim::partition_three_way(void *temporary_storage, size_t &storage_size, InputIterator input, FirstOutputIterator output_first_part, SecondOutputIterator output_second_part, UnselectedOutputIterator output_unselected, SelectedCountOutputIterator selected_count_output, const size_t size, FirstUnaryPredicate select_first_part_op, SecondUnaryPredicate select_second_part_op, const hipStream_t stream = 0, const bool debug_synchronous = false)
