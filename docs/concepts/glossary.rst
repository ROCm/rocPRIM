.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _glossary:

********************************************************************
 Glossary
********************************************************************

This glossary is to help users understand the basic concepts or terminologies used in the rocPRIM library.

.. glossary::

   Warp
       Refers to a group of threads that execute in single instruction, multiple thread (SIMT) fashion. Also known as wavefronts on AMD GPUs.

   Hardware Warp Size
       Refers to the number of threads in a warp defined by the hardware. On Nvidia GPUs a warp size is 32, while on AMD GPUs a warp size is 64.

   Logical Warp Size
       Refers to the number of threads in a warp defined by the user, which can be equal to or less than the size of the hardware warp size.

   Lane ID
       Refers to the thread identifier within the warp. A logical lane ID refers to the thread identifier in a "logical
       warp", which can be smaller than a hardware warp size (and can be defined as ``lane_id() % WarpSize``).

   Warp ID
      Refers to the identifier of the hardware/logical warp in a block. Warp ID is guaranteed to be unique among warps.

   Block
      Refers to a group of threads that are executed on the same compute unit (streaming multiprocessor). These threads can 
      be indexed using 1 Dimension {X}, 2 Dimensions {X, Y} or 3 Dimensions {X, Y, Z}. A block consists of multiple warps.

   Tile
      Refers to a block in C++AMP/HIPCC nomenclature.

   Flat ID
      Refers to a flattened identifier of a block (tile) or a thread identifier. Flat ID is a 1D value created from 2D or 3D 
      identifier. For example the flat ID of thread ID (X, Y) in 2D thread block 128x4 (XxY) is ``Y * 128 + X``.
