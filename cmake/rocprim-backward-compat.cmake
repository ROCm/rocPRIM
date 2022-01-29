# ########################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc.
# ########################################################################
cmake_minimum_required(VERSION 3.16.8)

#Include Directory Paths for - wrapper.
set(PROJECT_WRAPPER_TEMPLATE_FILE       ${PROJECT_SOURCE_DIR}/header.hpp.in)
set(PROJECT_BUILD_WRAPPER_ROOT_DIR     	${PROJECT_BINARY_DIR}/rocprim/wrapper/include/rocprim)

#Source Include File Directories, Level1 and Level2 Subdirectories
set(PROJECT_SOURCE_INCLUDE_ROOT_DIR     ${PROJECT_SOURCE_DIR}/rocprim/include/rocprim)

#Relative Path for wrapper (include) files generated 
set(PROJECT_WRAPPER_RELPATH_PREFIX    	"../../include/rocprim")

#Include Guard for wrapper (include) files generated
set(INCLUDE_GUARD_PREFIX 		"ROCPRIM_WRAPPER")

# Function for Generating Wrapper Headers for Backward Compatibilty
# Generates wrapper for all *.hpp files in include/rocprim folder and subfolders
# No Arguments to function.
# Wrapper are generated under rocprim/include/rocprim install folder and subfolders.
function(package_header_wrappers)
    file(GLOB_RECURSE include_file_names "${PROJECT_SOURCE_INCLUDE_ROOT_DIR}/*.hpp" )
    foreach(include_file_name ${include_file_names})
	file( RELATIVE_PATH sub_path_from_root ${PROJECT_SOURCE_INCLUDE_ROOT_DIR} ${include_file_name} )
	set( WRAPPER_FILE_NAME "${PROJECT_BUILD_WRAPPER_ROOT_DIR}/${sub_path_from_root}" )
	set( include_rel_path "${PROJECT_WRAPPER_RELPATH_PREFIX}/${sub_path_from_root}" )
	string(REPLACE "/" ";" sub_path_dirs "${sub_path_from_root}")
	foreach(subdir ${sub_path_dirs})
		if(NOT (subdir STREQUAL '' OR subdir STREQUAL '.'))
			set(include_rel_path "../${include_rel_path}")
		endif()
        endforeach()
	set(include_statements "\"${include_rel_path}\"\n")
        string(TOUPPER ${sub_path_from_root} guard_uc)
	string(REPLACE "." "_" guard_postfix "${guard_uc}" )
	string(REPLACE "/" "_" guard_postfix "${guard_postfix}" )
	set(include_guard "${INCLUDE_GUARD_PREFIX}_${guard_postfix}")
	configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${WRAPPER_FILE_NAME}")
    endforeach()
endfunction()


# Function for Generating Wrapper Header for Backward Compatibilty
# Generates wrapper for gen_file_name given as input 
# gen_file_name -  Arguments to function (absolute file name) to give as input file for which wrapper header is generated
# Wrapper generated for the input file under rocprim/include/rocprim/ install folder.
function (package_single_headerfile_wrapper gen_file_name)
	set(include_file ${gen_file_name})
	get_filename_component( file_name ${include_file} NAME)
	get_filename_component(file_name_we ${include_file} NAME_WE)
	string(TOUPPER ${file_name_we} INC_FILE_NAME)
	set(include_guard "${INCLUDE_GUARD_PREFIX}_${INC_FILE_NAME}_HPP")
	set(include_statements "\"../${PROJECT_WRAPPER_RELPATH_PREFIX}/${file_name}\"\n")
	configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_BUILD_WRAPPER_ROOT_DIR}/${file_name}")
	unset(include_guard)
	unset(include_statements)
endfunction()

