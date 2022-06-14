Warp-Level Operations
=====================

Exchange
--------

.. doxygenclass:: warp_exchange

Load
----

Class
.....

.. doxygenclass:: warp_load

Algorithms
..........

.. doxygenenum:: warp_load_method


Store
-----

Class
.....

.. doxygenclass:: warp_store

Algorithms
..........

.. doxygenenum:: warp_store_method


Reduce
------

.. doxygenclass:: warp_reduce

Scan
----

.. doxygenclass:: warp_scan

Sort
----

.. doxygenclass:: warp_sort

Shuffle
-------

.. doxygenfunction:: warp_shuffle (const T &input, const int src_lane, const int width)
.. doxygenfunction:: warp_shuffle_down (const T &input, const unsigned int delta, const int width)
.. doxygenfunction:: warp_shuffle_xor (const T &input, const int lane_mask, const int width)
