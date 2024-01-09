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
      - AMD Supports
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
