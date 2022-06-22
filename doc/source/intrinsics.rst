Intrinsics
==========

Bitwise
-------

.. doxygenfunction:: rocprim::get_bit(int x, int i)
.. doxygenfunction:: rocprim::bit_count(unsigned int x)
.. doxygenfunction:: rocprim::bit_count(unsigned long long x)

Warp size
---------

.. doxygenfunction:: rocprim::warp_size()
.. doxygenfunction:: rocprim::host_warp_size()
.. doxygenfunction:: rocprim::device_warp_size()

Lane and Warp ID
----------------

.. doxygenfunction:: lane_id ()

.. doxygenfunction:: warp_id (unsigned int flat_id)

..
   issue with breathe with template function

   .. doxygenfunction:: warp_id ()
   .. doxygenfunction:: template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ> warp_id ()

.. cpp:function:: unsigned int warp_id()

   Returns warp id in a block (tile).

.. cpp:function:: template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ> warp_id ()

   Returns warp id in a block (tile). Use template parameters to
   optimize 1D or 2D kernels.

Flat ID
-------

.. doxygenfunction:: flat_block_thread_id() -> typename std::enable_if<(BlockSizeY == 1 && BlockSizeZ == 1), unsigned int>::type
.. doxygenfunction:: flat_block_thread_id ()
.. doxygenfunction:: flat_block_id ()
.. doxygenfunction:: flat_block_id () -> typename std::enable_if<(BlockSizeY==1 &&BlockSizeZ==1), unsigned int >::type

Flat Size
---------

.. doxygenfunction:: rocprim::flat_block_size()
.. doxygenfunction:: rocprim::flat_tile_size()

Synchronization
---------------

.. doxygenfunction:: rocprim::syncthreads()
.. doxygenfunction:: rocprim::wave_barrier()

Active threads
--------------


.. doxygenfunction:: rocprim::ballot (int predicate)
.. doxygenfunction:: rocprim::masked_bit_count (lane_mask_type x, unsigned int add=0)
