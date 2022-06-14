Scan
----

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

scan
....

.. doxygenstruct:: scan_config

scan_by_key
...........

.. doxygenstruct:: scan_by_key_config

scan
~~~~

inclusive
.........

.. doxygenfunction:: inclusive_scan (void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, const size_t size, BinaryFunction scan_op=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)

exclusive
.........

.. doxygenfunction:: exclusive_scan (void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, const InitValueType initial_value, const size_t size, BinaryFunction scan_op=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)

segmented, inclusive
....................

.. doxygenfunction:: segmented_inclusive_scan (void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, BinaryFunction scan_op=BinaryFunction(), hipStream_t stream=0, bool debug_synchronous=false)

segmented, exclusive
....................

.. doxygenfunction:: segmented_exclusive_scan (void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, const InitValueType initial_value, BinaryFunction scan_op=BinaryFunction(), hipStream_t stream=0, bool debug_synchronous=false)x

scan_by_key
~~~~~~~~~~~

inclusive
.........

.. doxygenfunction:: inclusive_scan_by_key (void *const temporary_storage, size_t &storage_size, const KeysInputIterator keys_input, const ValuesInputIterator values_input, const ValuesOutputIterator values_output, const size_t size, const BinaryFunction scan_op=BinaryFunction(), const KeyCompareFunction key_compare_op=KeyCompareFunction(), const hipStream_t stream=0, const bool debug_synchronous=false)

exclusive
.........

.. doxygenfunction:: exclusive_scan_by_key (void *const temporary_storage, size_t &storage_size, const KeysInputIterator keys_input, const ValuesInputIterator values_input, const ValuesOutputIterator values_output, const InitialValueType initial_value, const size_t size, const BinaryFunction scan_op=BinaryFunction(), const KeyCompareFunction key_compare_op=KeyCompareFunction(), const hipStream_t stream=0, const bool debug_synchronous=false)
