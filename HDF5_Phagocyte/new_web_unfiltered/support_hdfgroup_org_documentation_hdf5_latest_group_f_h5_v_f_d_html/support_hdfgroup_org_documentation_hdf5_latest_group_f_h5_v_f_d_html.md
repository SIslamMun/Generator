Help us improve by taking our short survey: <https://www.hdfgroup.org/website-survey/>
![Logo](https://support.hdfgroup.org/documentation/hdf5/latest/HDFG-logo.png) |  HDF5 Last Updated on 2026-01-10 The HDF5 Field Guide  
---|---  
  * [Main Page](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
  * [Getting started](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * [User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * [Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
  * [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * [RFCs](https://support.hdfgroup.org/documentation/hdf5/latest/_r_f_c.html)
  * [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
  * [Glossary](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html)
  * [Full-Text Search](https://support.hdfgroup.org/documentation/hdf5/latest/_f_t_s.html)
  * [About](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html)
  * [![](https://support.hdfgroup.org/documentation/hdf5/latest/search/close.svg)](javascript:searchBox.CloseResultsWindow\(\))


### Table of contents
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#title0 "H2")
  * [ Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#title1 "H2")
  * [ Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#title2 "H2")
  * [Function/Subroutine Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#title3 "H2")
  * [◆ h5fdsubfiling_get_file_mapping_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#namespaces) | [Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#func-members)
Fortran VFD (H5VFD) Interface
## Detailed Description 

See also
     [Virtual File Driver Features](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html), C-API 
##  Modules  
---  
module  | [h5fd](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fd.html)  
| This module contains Fortran interfaces for H5VFD (Virtual File Driver) functions.   
  
##  Functions/Subroutines  
---  
subroutine  |  [h5fdsubfiling_get_file_mapping_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#ga6dcf085c6a97d0e9aeb5bf53e17221ec) (file_id, filenames, num_files, hdferr)  
| Retrieve the list of subfile names for a HDF5 file for the subfiling VFD.   
  
## Function/Subroutine Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html#ga6dcf085c6a97d0e9aeb5bf53e17221ec)h5fdsubfiling_get_file_mapping_f()
subroutine h5fdsubfiling_get_file_mapping_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _file_id_ ,   
---|---|---|---  
|  | character(len=default_max_len), dimension(:), intent(out), allocatable |  _filenames_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(out) |  _num_files_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Retrieve the list of subfile names for a HDF5 file for the subfiling VFD. 
This function retrieves the names of all subfiles associated with an HDF5 file that uses the subfiling Virtual File Driver (VFD). The subfiling VFD distributes file data across multiple subfiles to improve parallel I/O performance on shared file systems.
The returned filenames correspond to the physical subfiles stored on the file system that collectively make up the logical HDF5 file. This information is useful for:
  * File management and backup operations
  * Understanding the physical storage layout
  * Tools like h5fuse for recombining subfiles
  * Debugging subfiling configurations



Parameters
     file_id | [in] HDF5 file identifier for a file using the subfiling VFD   
---|---  
filenames | [out] Allocatable array of subfile names. Memory is automatically allocated by the function and must be deallocated by the caller. See **Compiler Compatibility** note below.   
num_files | [out] Number of subfiles in the filenames array   
hdferr | [out] Error code: 
  * 0 on success 
  * -1 on failure



Since
    2.0.0 

Note
     **Memory Management** : The filenames array is automatically allocated. The caller is responsible for deallocating when it is no longer needed:   
DEALLOCATE(filenames)      **Compiler Compatibility** :
  * With H5_FORTRAN_HAVE_CHAR_ALLOC: Variable-length character strings (Fortran 2003+)
  * Without H5_FORTRAN_HAVE_CHAR_ALLOC: Fixed-length 8192 character strings (older compilers), filenames longer than 8192 characters will cause the function to fail with hdferr = -1. 



Note
    This function will not be accessible if support for the subfiling VFD is unavailable or disabled.  

Note
     **Optimized Allocation** : The function uses knowledge of the subfiling filename template to estimate optimal string lengths, typically reducing memory usage compared to the maximum 8192 character limit. Subfile names follow the pattern: `basename.subfile_<inode>_<index>_of_<total>` 

Example Usage:
    
USE [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
IMPLICIT NONE
INTEGER(HID_T) :: file_id
CHARACTER(LEN=:), ALLOCATABLE, DIMENSION(:) :: subfile_names
INTEGER(SIZE_T) :: num_subfiles
INTEGER :: hdferr
INTEGER :: i
! Open file with subfiling VFD (file_id setup not shown)
! Get subfile mapping
CALL h5fdsubfiling_get_file_mapping_f(file_id, subfile_names, num_subfiles, hdferr)
IF (hdferr == 0) THEN
print *, 'Found', num_subfiles, 'subfiles:'
DO i = 1, num_subfiles
print *, ' ', trim(subfile_names(i))
END DO
! Clean up
DEALLOCATE(subfile_names)
ELSE
print *, 'Error getting file mapping:', hdferr
END IF
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26 

See also
     [H5FDsubfiling_get_file_mapping()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7 "Retrieve the subfile names for an HDF5 file using the subfiling VFD.") (C API)       H5Pset_fapl_subfiling_f() for setting up subfiling VFD 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


