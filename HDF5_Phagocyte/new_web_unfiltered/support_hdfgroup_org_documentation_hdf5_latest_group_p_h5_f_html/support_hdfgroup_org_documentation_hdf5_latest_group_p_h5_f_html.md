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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#title2 "H2")
  * [◆ H5Fget_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#title3 "H2")
  * [◆ H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#func-members)
Parallel
[Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, bool *flag)  
| Retrieves the atomicity mode in use.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fset_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, bool flag)  
| Sets the MPI atomicity mode.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd)H5Fget_mpi_atomicity()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mpi_atomicity  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | bool * |  _flag_ )  
Retrieves the atomicity mode in use.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | flag | Logical flag for atomicity setting. Valid values are: 
  * 1 – MPI file access is set to atomic mode. 
  * 0 – MPI file access is set to nonatomic mode. 



Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fget_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd "Retrieves the atomicity mode in use.") retrieves the current consistency semantics mode for data access for the file `file_id`.
Upon successful return, `flag` will be set to `1` if file access is set to atomic mode and `0` if file access is set to nonatomic mode. 

See also
    [Enabling a Strict Consistency Semantics Model in Parallel HDF5](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC%20PHDF5%20Consistency%20Semantics%20MC%20120328.docx.pdf) 

Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359)H5Fset_mpi_atomicity()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fset_mpi_atomicity  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | bool |  _flag_ )  
Sets the MPI atomicity mode.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[in] | flag | Logical flag for atomicity setting. Valid values are: 
  * `1` – Sets MPI file access to atomic mode. 
  * `0` – Sets MPI file access to nonatomic mode. 



Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Motivation
     [H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") is applicable only in parallel environments using MPI I/O. The function is one of the tools used to ensure sequential consistency. This means that a set of operations will behave as though they were performed in a serial order consistent with the program order.  
[H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") sets MPI consistency semantics for data access to the file, `file_id`.
If `flag` is set to `1`, all file access operations will appear atomic, guaranteeing sequential consistency. If `flag` is set to `0`, enforcement of atomic file access will be turned off.
[H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") is a collective function and all participating processes must pass the same values for `file_id` and `flag`.
This function is available only when the HDF5 library is configured with parallel support (`HDF5_ENABLE_PARALLEL`). It is useful only when used with the [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) driver (see [H5Pset_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")).  

Attention
    
[H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") calls `MPI_File_set_atomicity` underneath and is not supported if the execution platform does not support `MPI_File_set_atomicity`. When it is supported and used, the performance of data access operations may drop significantly.
In certain scenarios, even when `MPI_File_set_atomicity` is supported, setting atomicity with [H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") and `flag` set to 1 does not always yield strictly atomic updates. For example, some [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") calls translate to multiple `MPI_File_write_at` calls. This happens in all cases where the high-level file access routine translates to multiple lower level file access routines. The following scenarios will raise this issue: 
  * Non-contiguous file access using independent I/O 
  * Partial collective I/O using chunked access 
  * Collective I/O using filters or when data conversion is required


This issue arises because MPI atomicity is a matter of MPI file access operations rather than HDF5 access operations. But the user is normally seeking atomicity at the HDF5 level. To accomplish this, the application must set a barrier after a write, [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."), but before the next read, [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer."), in addition to calling [H5Fset_mpi_atomicity()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.").The barrier will guarantee that all underlying write operations execute atomically before the read operations starts. This ensures additional ordering semantics and will normally produce the desired behavior.  

See also
    [Enabling a Strict Consistency Semantics Model in Parallel HDF5](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC%20PHDF5%20Consistency%20Semantics%20MC%20120328.docx.pdf) 

Since
    1.8.9 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


