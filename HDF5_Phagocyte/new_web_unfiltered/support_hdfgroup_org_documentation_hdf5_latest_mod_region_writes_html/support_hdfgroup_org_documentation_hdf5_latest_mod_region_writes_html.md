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
  * [ Introduction to Modified Region Writes](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title0 "H1")
  * [ How the Core VFD Tracks File Modifications](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title1 "H2")
  * [ Using the New Feature](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title2 "H2")
  * [ Performance](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title3 "H2")
  * [ References](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title4 "H1")
  * [ The Virtual File Layer and Virtual File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/mod_region_writes.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Modified Region Writes
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
* * *
The Core virtual file driver allows the manipulating of HDF5 files in memory instead of in physical storage. In previous versions, changing any part of a file in memory meant the entire file would be written to storage on file close or flush. To improve the performance of the writing to storage operation, a new feature, modified region writes, has been added. With modified region writes, only the changed regions of the file are written to storage.
Introduced with HDF5-1.8.13 May 15, 2014
The intended audience for this feature is advanced users of the Core virtual file driver.
#  Introduction to Modified Region Writes
In the 1.8.13 release of the HDF5 Library, a feature called modified region writes was added to improve the performance of writes to storage. The purpose of this document is to describe the feature and how to use it. The intended audience for this feature is advanced users of the Core virtual file driver (VFD). The Core (or Memory) VFD allows HDF5 files to be created or opened in memory instead of in physical storage. If an existing file is opened in memory, the entire contents of the file are copied into memory on open. All subsequent manipulations of created or opened files occur in memory. The advantage of working on files in memory is the file operations go much faster, but the disadvantage is significant memory resources may be required when working with large files. On file close or flush, the changes can optionally be propagated to physical storage.
The Core VFD is configured via the following API call: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t increment, bool backing_store)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f)
herr_t H5Pset_fapl_core(hid_t fapl_id, size_t increment, bool backing_store)
Modifies the file access property list to use the H5FD_CORE driver.
The backing_store parameter sets whether or not changes are propagated to physical storage on close. If this parameter is set to 0 (FALSE), then all changes will be lost when the file is closed. If set to 1 (TRUE), then the changes are written to storage on file close or flush. In previous versions of the library when a file was closed, the entire file would be written out if even a single byte has changed. This can be inefficient when very large files are written out after minimal changes have been made.
If files being worked on in memory will be written to disk, the modified region writes feature can be enabled.
##  How the Core VFD Tracks File Modifications
When modified region writes are enabled, the Core VFD will track any changes made to the file. On file close or flush, the tracked changes will be written to storage.
As write calls pass through the Core VFD, a list of “start address-end address” pairs representing the writes is updated. This list serves as a map of modified regions in the file. Overlapping or abutting regions are merged as they are inserted into the list.
As a further optimization, a write page size can be set. This feature expands any dirty regions (regions with changed bytes) to the nearest page boundaries. Using write pages can minimize seeks and small, inefficient writes when a large number of small non-adjacent writes occur. See the figure below.
Note that these marked regions are at the granularity of the write calls that the library makes. In other words, an entire metadata object or dataset chunk will be marked dirty if even a single byte is changed since the library uses a single write call when metadata objects or dataset chunks are evicted from their respective caches. The Core VFD will make no effort to determine the particular bytes that were modified with respect to the original data.
![](https://support.hdfgroup.org/documentation/hdf5/latest/modregwrite.png)  
---  
##  Using the New Feature
The modified region writes feature is turned off by default. Setting the **backing_store** flag to TRUE will not turn modified region writes on.
The modified region writes feature is controlled via the [H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469 "Gets information about the write tracking feature used by the core VFD.")/[H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250 "Sets write tracking information for core driver, H5FD_CORE.") HDF5 API calls. The signatures of these function calls are the following: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool is_enabled, size_t page_size)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool *is_enabled, size_t *page_size)
[H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469)
herr_t H5Pget_core_write_tracking(hid_t fapl_id, bool *is_enabled, size_t *page_size)
Gets information about the write tracking feature used by the core VFD.
[H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250)
herr_t H5Pset_core_write_tracking(hid_t fapl_id, bool is_enabled, size_t page_size)
Sets write tracking information for core driver, H5FD_CORE.
Setting the page size to a value greater than 1 turns write tracking on at that page size. Setting a page size of 1 byte disables paging.
More information for these function calls can be found in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
##  Performance
The performance benefits of the feature will depend heavily on the data access patterns of the application and will have to be evaluated on a case-by-case basis. In cases where the majority of the data would be written out (for example, creating and writing data to a new file), the new feature will likely not impart a significant performance benefit. In cases where a small amount of data will be added or changed (for example, opening an existing file and modifying a small amount of existing data), the performance benefits could be significant.
When performance tuning, the following parameters are likely to have significant effects on I/O throughput: 
  * The size of the backing store pages (see [H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250 "Sets write tracking information for core driver, H5FD_CORE.")) 
  * Dataset layout and chunk size (see [H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset.") and H5Pset_chunk) 
  * Metadata aggregation size (see [H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.")) 
  * Using the latest file format (see [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.")) 
  * Data layout considerations (arrangement of groups, datasets, and datatypes)


In general, anything that promotes the aggregation of changes made to the file will enhance the performance of this feature. Unfortunately, empirical testing will typically be required to determine the “sweet spot” between reducing the number of seeks and minimizing the amount of data written out.
More information for these function calls can be found in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
#  References
For more information, see the entries for the [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver."), [H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469 "Gets information about the write tracking feature used by the core VFD."), and [H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250 "Sets write tracking information for core driver, H5FD_CORE.") function calls in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
##  The Virtual File Layer and Virtual File Drivers
The HDF5 Library uses a layered architecture. The lowest layer is the virtual file layer (VFL). The VFL handles low-level file I/O via virtual file drivers (VFDs). The VFL is an abstraction layer in the HDF5 Library that maps I/O operations such as “read” to concrete I/O calls like the POSIX read() call or the Win32 ReadFile() call. Each VFD implements a different I/O scheme: some examples are MPI-I/O, POSIX I/O, and in-memory I/O. This VFL/VFD scheme allows abstract HDF5 file manipulations to be separated from storage I/O operations.
For more information, see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html)
For more information on virtual file drivers, see the [Alternate File Storage Layouts and Low-level File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_alternate_drivers) section in the [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file) chapter in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html).
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


