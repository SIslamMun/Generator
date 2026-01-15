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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#title2 "H2")
  * [◆ H5Fget_metadata_read_retry_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#title3 "H2")
  * [◆ H5Fstart_swmr_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#func-members)
Single Writer Multiple Readers
[Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_metadata_read_retry_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#gaa80bd62f19993e414e383db7d1623e5f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [H5F_retry_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html) *info)  
| Retrieves the collection of read retries for metadata entries with checksum.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id)  
| Retrieves free-space section information for a file.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#gaa80bd62f19993e414e383db7d1623e5f)H5Fget_metadata_read_retry_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_metadata_read_retry_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  |  [H5F_retry_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html) * |  _info_ )  
Retrieves the collection of read retries for metadata entries with checksum.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | info | Struct containing the collection of read retries for metadata entries with checksum  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
  
**Failure Modes:**
  * When the input identifier is not a file identifier. 
  * When the pointer to the output structure is NULL. 
  * When the memory allocation for `retries` failed.


[H5Fget_metadata_read_retry_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#gaa80bd62f19993e414e383db7d1623e5f "Retrieves the collection of read retries for metadata entries with checksum.") retrieves information regarding the number of read retries for metadata entries with checksum for the file `file_id`. This information is reported in the [H5F_retry_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html) struct defined in [H5Fpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html) as follows: 
#define H5F_NUM_METADATA_READ_RETRY_TYPES 21
typedef struct [H5F_retry_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html) {
unsigned [nbins](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html#aa86f48adcf8ee5bbe9d05ae595130544);
uint32_t *[retries](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html#a9ed64b2becff4cade9713b0cd9f3827e)[[H5F_NUM_METADATA_READ_RETRY_TYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a3a67d0d121762e7c44c4ef3f40a49611)];
} [H5F_retry_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__retry__info__t.html);
`nbins` is the number of bins for each `retries`[i] of metadata entry `i`. It is calculated based on the current number of read attempts used in the library and logarithmic base 10.
If read retries are incurred for a metadata entry `i`, the library will allocate memory for `retries[i] (nbins * sizeof(uint32_t)` and store the collection of retries there. If there are no retries for a metadata entry `i`, `retries[i]` will be NULL. After a call to this routine, users should free each `retries[i]` that is non-NULL, otherwise resource leak will occur.
For the library default read attempts of 100 for SWMR access, nbins will be 2 as depicted below: 
  * `retries[i][0]` is the number of 1 to 9 read retries. 
  * `retries[i][1]` is the number of 10 to 99 read retries. For the library default read attempts of 1 for non-SWMR access, `nbins` will be 0 and each `retries[i]` will be NULL.


The following table lists the 21 metadata entries of `retries[]`: 
Index for `retries[]` | Metadata entries*  
---|---  
0 | Object header (version 2)   
1 | Object header chunk (version 2)   
2 | B-tree header (version 2)   
3 | B-tree internal node (version 2)   
4 | B-tree leaf node (version 2)   
5 | Fractal heap header   
6 | Fractal heap direct block (optional checksum)   
7 | Fractal heap indirect block   
8 | Free-space header   
9 | Free-space sections   
10 | Shared object header message table   
11 | Shared message record list   
12 | Extensive array header   
13 | Extensive array index block   
14 | Extensive array super block   
15 | Extensive array data block   
16 | Extensive array data block page   
17 | Fixed array super block   
18 | Fixed array data block   
19 | Fixed array data block page   
20 | File's superblock (version 2)   
* All entries are of version 0 (zero) unless indicated otherwise.  

Note
    On a system that is not atomic, the library might possibly read inconsistent metadata with checksum when performing single-writer/multiple-reader (SWMR) operations for an HDF5 file. Upon encountering such situations, the library will try reading the metadata again for a set number of times to attempt to obtain consistent data. The maximum number of read attempts used by the library will be either the value set via [H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list.") or the library default value when a value is not set.  
When the current number of metadata read attempts used in the library is unable to remedy the reading of inconsistent metadata on a system, the user can assess the information obtained via this routine to derive a different maximum value. The information can also be helpful for debugging purposes to identify potential issues with metadata flush dependencies and SWMR implementation in general. 

Since
    1.10.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe)H5Fstart_swmr_write()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fstart_swmr_write  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _file_id_ | ) |   
---|---|---|---|---|---  
Retrieves free-space section information for a file.  

Parameters
     [in] | file_id | File identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fstart_swmr_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe "Retrieves free-space section information for a file.") will activate SWMR writing mode for a file associated with `file_id`. This routine will prepare and ensure the file is safe for SWMR writing as follows: 
  * Check that the file is opened with write access ([H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)). 
  * Check that the file is opened with the latest library format to ensure data structures with check-summed metadata are used. 
  * Check that the file is not already marked in SWMR writing mode. 
  * Enable reading retries for check-summed metadata to remedy possible checksum failures from reading inconsistent metadata on a system that is not atomic. 
  * Turn off usage of the library's accumulator to avoid possible ordering problem on a system that is not atomic. 
  * Perform a flush of the file's data buffers and metadata to set a consistent state for starting SWMR write operations.


Library objects are groups, datasets, and committed datatypes. For the current implementation, groups and datasets can remain open when activating SWMR writing mode, but not committed datatypes. Attributes attached to objects cannot remain open either. 

Since
    1.10.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


