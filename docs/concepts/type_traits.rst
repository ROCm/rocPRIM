.. meta::
  :description: Implementing traits for custom types in rocPRIM
  :keywords: rocPRIM, ROCm, custom types, type traits

.. _type_traits:

********************************************************************
 Implementing traits for custom types in rocPRIM
********************************************************************

Overview
========

This interface is designed to enable users to provide additional type trait information to rocPRIM, facilitating better compatibility with custom types.

Accurately describing custom types is important for performance optimization and computational correctness.

Custom types that implement arithmetic operators can behave like built-in arithmetic types but might still be interpreted by rocPRIM algorithms as generic ``struct`` or ``class`` types.

The rocPRIM type traits interface lets users add custom trait information for their types, improving compatibility between these types and rocPRIM algorithms.

This interface is similar to operator overloading.

Traits should be implemented as required by specific algorithms. Some traits can't be defined if they can be inferred from others.

Interface
=========

.. doxygengroup::  type_traits_interfaces
  :content-only:
  :members:

Available traits
================

.. doxygengroup::  available_traits
  :content-only:
  :members:

Type traits wrappers
====================
.. doxygengroup::  rocprim_type_traits_wrapper
  :content-only:
  :no-link:

Types with predefined traits
============================

.. doxygengroup::  rocprim_pre_defined_traits
  :content-only:
  :members:
  :outline:
  :no-link:


