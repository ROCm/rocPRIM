# ########################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc.
# ########################################################################
cmake_minimum_required(VERSION 3.16.8)

#Include Directory Paths for - wrapper.
set(PROJECT_WRAPPER_TEMPLATE_FILE       ${PROJECT_SOURCE_DIR}/header.hpp.in)
set(PROJECT_WRAPPER_MAIN_INC_DIR        ${PROJECT_BINARY_DIR}/include/rocprim)
set(PROJECT_WRAPPER_INC_BLOCK           ${PROJECT_BINARY_DIR}/include/rocprim/block)
set(PROJECT_WRAPPER_INC_DETAIL          ${PROJECT_BINARY_DIR}/include/rocprim/detail)
set(PROJECT_WRAPPER_INC_DEVICE          ${PROJECT_BINARY_DIR}/include/rocprim/device)
set(PROJECT_WRAPPER_INC_INTRINSICS      ${PROJECT_BINARY_DIR}/include/rocprim/intrinsics)
set(PROJECT_WRAPPER_INC_ITERATOR        ${PROJECT_BINARY_DIR}/include/rocprim/iterator)
set(PROJECT_WRAPPER_INC_TYPES           ${PROJECT_BINARY_DIR}/include/rocprim/types)
set(PROJECT_WRAPPER_INC_THREAD          ${PROJECT_BINARY_DIR}/include/rocprim/thread)
set(PROJECT_WRAPPER_INC_WARP            ${PROJECT_BINARY_DIR}/include/rocprim/warp)
set(PROJECT_WRAPPER_INC_BLK_DET         ${PROJECT_BINARY_DIR}/include/rocprim/block/detail)
set(PROJECT_WRAPPER_INC_DEV_DET         ${PROJECT_BINARY_DIR}/include/rocprim/device/detail)
set(PROJECT_WRAPPER_INC_DEV_SPE         ${PROJECT_BINARY_DIR}/include/rocprim/device/specialization)
set(PROJECT_WRAPPER_INC_ITE_DET         ${PROJECT_BINARY_DIR}/include/rocprim/iterator/detail)
set(PROJECT_WRAPPER_INC_WAR_DET         ${PROJECT_BINARY_DIR}/include/rocprim/warp/detail)


#Source Include File Directories, Level1 and Level2 Subdirectories
set(PROJECT_FREORG_MAIN_INC_DIR         ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim)
set(PROJECT_FREORG_INC_BLOCK            ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/block)
set(PROJECT_FREORG_INC_DETAIL           ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/detail)
set(PROJECT_FREORG_INC_DEVICE           ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/device)
set(PROJECT_FREORG_INC_INTRINSICS       ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/intrinsics)
set(PROJECT_FREORG_INC_ITERATOR         ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/iterator)
set(PROJECT_FREORG_INC_TYPES            ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/types)
set(PROJECT_FREORG_INC_THREAD           ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/thread)
set(PROJECT_FREORG_INC_WARP             ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/warp)
set(PROJECT_FREORG_INC_BLK_DET          ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/block/detail)
set(PROJECT_FREORG_INC_DEV_DET          ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/device/detail)
set(PROJECT_FREORG_INC_DEV_SPE          ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/device/specialization)
set(PROJECT_FREORG_INC_ITE_DET          ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/iterator/detail)
set(PROJECT_FREORG_INC_WAR_DET          ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim/warp/detail)

#Relative Path for wrapper (include) files generated 
set(PROJECT_WRAPPER_MAIN_INC_RELPATH    "${include_statements}#include \"../../../include/rocprim")
set(PROJECT_WRAPPER_BLOCK_RELPATH       "${include_statements}#include \"../../../../include/rocprim/block")
set(PROJECT_WRAPPER_DETAIL_RELPATH      "${include_statements}#include \"../../../../include/rocprim/detail")
set(PROJECT_WRAPPER_DEVICE_RELPATH      "${include_statements}#include \"../../../../include/rocprim/device")
set(PROJECT_WRAPPER_INTRINSICS_RELPATH  "${include_statements}#include \"../../../../include/rocprim/intrinsics")
set(PROJECT_WRAPPER_ITERATOR_RELPATH    "${include_statements}#include \"../../../../include/rocprim/iterator")
set(PROJECT_WRAPPER_TYPES_RELPATH       "${include_statements}#include \"../../../../include/rocprim/types")
set(PROJECT_WRAPPER_THREAD_RELPATH      "${include_statements}#include \"../../../../include/rocprim/thread")
set(PROJECT_WRAPPER_WARP_RELPATH        "${include_statements}#include \"../../../../include/rocprim/warp")
set(PROJECT_WRAPPER_BLK_DET_RELPATH     "${include_statements}#include \"../../../../../include/rocprim/block/detail")
set(PROJECT_WRAPPER_DEV_DET_RELPATH     "${include_statements}#include \"../../../../../include/rocprim/device/detail")
set(PROJECT_WRAPPER_DEV_SPE_RELPATH     "${include_statements}#include \"../../../../../include/rocprim/device/specialization")
set(PROJECT_WRAPPER_ITE_DET_RELPATH     "${include_statements}#include \"../../../../../include/rocprim/iterator/detail")
set(PROJECT_WRAPPER_WAR_DET_RELPATH     "${include_statements}#include \"../../../../../include/rocprim/warp/detail")


# Function for Generating Wrapper Headers for Backward Compatibilty
# Generates wrapper for all *.h files in include/rocprim folder
# No Arguments to function.
# Wrapper are generated under rocprim/include/rocprim.
function (package_gen_bkwdcomp_hdrs)
	# Get list of *.h files in folder include/rocprim 
	file(GLOB include_files ${PROJECT_FREORG_MAIN_INC_DIR}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_MAIN_INC_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_MAIN_INC_DIR}/${file_name}")
		unset(include_statements)
	endforeach()

	# Get list of *.h files in folder include/rocprim/block 
	file(GLOB include_files ${PROJECT_FREORG_INC_BLOCK}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_BLOCK_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_BLOCK}/${file_name}")
		unset(include_statements)
	endforeach()

	# Get list of *.h files in folder include/rocprim/detail 
	file(GLOB include_files ${PROJECT_FREORG_INC_DETAIL}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_DETAIL_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DETAIL}/${file_name}")
		unset(include_statements)
	endforeach()

	# Get list of *.h files in folder include/rocprim/device 
	file(GLOB include_files ${PROJECT_FREORG_INC_DEVICE}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_DEVICE_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DEVICE}/${file_name}")
		unset(include_statements)
	endforeach()

	# Get list of *.h files in folder include/rocprim/intrinsics 
	file(GLOB include_files ${PROJECT_FREORG_INC_INTRINSICS}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_INTRINSICS_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_INTRINSICS}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/iterator 
	file(GLOB include_files ${PROJECT_FREORG_INC_ITERATOR}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_ITERATOR_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_ITERATOR}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/types 
	file(GLOB include_files ${PROJECT_FREORG_INC_TYPES}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_TYPES_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_TYPES}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/thread 
	file(GLOB include_files ${PROJECT_FREORG_INC_THREAD}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_THREAD_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_THREAD}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/warp 
	file(GLOB include_files ${PROJECT_FREORG_INC_WARP}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_WARP_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_WARP}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/block/detail
	file(GLOB include_files ${PROJECT_FREORG_INC_BLK_DET}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_BLK_DET_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_BLK_DET}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/device/detail 
	file(GLOB include_files ${PROJECT_FREORG_INC_DEV_DET}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_DEV_DET_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DEV_DET}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/device/specialization 
	file(GLOB include_files ${PROJECT_FREORG_INC_DEV_SPE}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_DEV_SPE_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DEV_SPE}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/iterator/detail 
	file(GLOB include_files ${PROJECT_FREORG_INC_ITE_DET}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_ITE_DET_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_ITE_DET}/${file_name}")
		unset(include_statements)
	endforeach()

        # Get list of *.h files in folder include/rocprim/warp/detail 
	file(GLOB include_files ${PROJECT_FREORG_INC_WAR_DET}/*.hpp)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_WAR_DET_RELPATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_WAR_DET}/${file_name}")
		unset(include_statements)
	endforeach()


endfunction()


# Function for Generating Wrapper Header for Backward Compatibilty
# Generates wrapper for gen_file_name given as input 
# gen_file_name -  Arguments to function (absolute file name) to give as input file for which wrapper header is generated
# Wrapper generated for the input file under rocprim/include/rocprim/.
function (package_gen_bkwdcomp_hdrfile gen_file_name)
        set(include_file ${gen_file_name})  
	get_filename_component( file_name ${include_file} NAME)
	set(include_statements "${PROJECT_WRAPPER_MAIN_INC_RELPATH}/${file_name}\"\n")
	configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_MAIN_INC_DIR}/${file_name}")
	unset(include_statements)
endfunction()

