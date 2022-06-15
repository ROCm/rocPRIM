Data movement functions
-----------------------

Direct Blocked
~~~~~~~~~~~~~~

Load
....

.. doxygenfunction:: block_load_direct_blocked(unsigned int flat_id, InputIterator block_input, T (&items)[ItemsPerThread])
.. doxygenfunction:: block_load_direct_blocked(unsigned int flat_id, InputIterator block_input, T (&items)[ItemsPerThread], unsigned int valid)
.. doxygenfunction:: block_load_direct_blocked (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread], unsigned int valid, Default out_of_bounds)

Store
.....

.. doxygenfunction:: block_store_direct_blocked (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread])
.. doxygenfunction:: block_store_direct_blocked (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread], unsigned int valid)

Direct Blocked Vectorized
~~~~~~~~~~~~~~~~~~~~~~~~~

Load
....

.. doxygenfunction:: block_load_direct_blocked_vectorized (unsigned int flat_id, T *block_input, U(&items)[ItemsPerThread])

Store
.....

.. doxygenfunction:: block_store_direct_blocked_vectorized (unsigned int flat_id, T *block_output, U(&items)[ItemsPerThread])

Direct Striped
~~~~~~~~~~~~~~

Load
....

.. doxygenfunction:: block_load_direct_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread])
.. doxygenfunction:: block_load_direct_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread], unsigned int valid)
.. doxygenfunction:: block_load_direct_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread], unsigned int valid, Default out_of_bounds)

Store
.....

.. doxygenfunction:: block_store_direct_striped (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread])
.. doxygenfunction:: block_store_direct_striped (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread], unsigned int valid)

Direct Warp Striped
~~~~~~~~~~~~~~~~~~~

Load
....

.. doxygenfunction:: block_load_direct_warp_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread])
.. doxygenfunction:: block_load_direct_warp_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread], unsigned int valid)
.. doxygenfunction:: block_load_direct_warp_striped (unsigned int flat_id, InputIterator block_input, T(&items)[ItemsPerThread], unsigned int valid, Default out_of_bounds)

Store
.....

.. doxygenfunction:: block_store_direct_warp_striped (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread])
.. doxygenfunction:: block_store_direct_warp_striped (unsigned int flat_id, OutputIterator block_output, T(&items)[ItemsPerThread], unsigned int valid)
