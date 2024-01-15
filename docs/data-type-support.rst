.. meta::
   :description: rocPRIM API library data type support
   :keywords: rocPRIM, ROCm, API library, API reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

* Supported input and output types.

  .. list-table:: Supported Input/Output Types
    :header-rows: 1
    :name: supported-input-output-types

    *
      - Input/Output Types
      - Library Data Type
      - Support
    *
      - int8
      - int8_t
      - 🟧
    *
      - float8
      - Not Supported
      - ❌
    *
      - bfloat8
      - Not Supported
      - ❌
    *
      - int16
      - int16_t
      - ❌
    *
      - float16
      - rocprim::half
      - 🟧
    *
      - bfloat16      
      - rocprim::bfloat16
      - 🟧
    *
      - int32
      - int
      - ✅
    *
      - tensorfloat32
      - Not Supported
      - ❌
    *
      - float32
      - float
      - ✅
    *
      - float64
      - double
      - ✅

The 🟧 means that the data type is mostly supported, but there are some API tests, that do not work.
* The ``block_histogram`` test fails with ``int8``.
* The ``device_histogram`` and ``device_reduce_by_key`` doesn't work with ``rocprim::half`` and ``rocprim::bfloat16``.
* The ``device_run_length_encode``, ``warp_exchange`` and ``warp_load`` doesn't work with ``rocprim::half``.
