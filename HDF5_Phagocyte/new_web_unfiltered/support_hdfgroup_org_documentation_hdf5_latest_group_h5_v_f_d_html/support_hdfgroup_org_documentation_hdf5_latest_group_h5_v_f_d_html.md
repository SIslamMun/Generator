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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title2 "H2")
  * [◆ H5FDdriver_query()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title3 "H2")
  * [◆ H5FDonion_get_revision_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title4 "H2")
  * [◆ H5FDsubfiling_get_file_mapping()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#func-members)
Virtual File Driver Features
[File Drivers (H5FD)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f_d.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5FDdriver_query](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga42cb42d302b233ce880a215784db0799) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) driver_id, unsigned long *flags)  
| Allows querying a VFD ID for features before the file is opened.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5FDonion_get_revision_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga54309eee8a5a936e3e56a2e20eba23f0) (const char *filename, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, uint64_t *revision_count)  
| get the number of revisions   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5FDsubfiling_get_file_mapping](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, char ***filenames, size_t *len)  
| Retrieve the subfile names for an HDF5 file using the subfiling VFD.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga42cb42d302b233ce880a215784db0799)H5FDdriver_query()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5FDdriver_query  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _driver_id_ ,   
---|---|---|---  
|  | unsigned long * |  _flags_ )  
Allows querying a VFD ID for features before the file is opened.  

Parameters
     [in] | driver_id | Virtual File Driver (VFD) ID   
---|---|---  
[out] | flags | VFD flags supported 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Queries a virtual file driver (VFD) for feature flags. Takes a VFD [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) so it can be used before the file is opened. For example, this could be used to check if a VFD supports SWMR. 

Note
    The flags obtained here are just those of the base driver and do not take any configuration options (e.g., set via a fapl call) into consideration. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga54309eee8a5a936e3e56a2e20eba23f0)H5FDonion_get_revision_count()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5FDonion_get_revision_count  | ( | const char * |  _filename_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
|  | uint64_t * |  _revision_count_ )  
get the number of revisions 
* * * 

Parameters
     [in] | filename | The name of the onion file   
---|---|---  
[in] | fapl_id | The ID of the file access property list   
[out] | revision_count | The number of revisions 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5FDonion_get_revision_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga54309eee8a5a936e3e56a2e20eba23f0 "get the number of revisions") returns the number of revisions for an onion file. It takes the file name and file access property list that is set for the onion VFD driver. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7)H5FDsubfiling_get_file_mapping()
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5FDsubfiling_get_file_mapping  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | char *** |  _filenames_ ,   
|  | size_t * |  _len_ )  
Retrieve the subfile names for an HDF5 file using the subfiling VFD.  

Parameters
     [in] | file_id | File identifier  
---|---|---  
[out] | filenames | Pointer to an array of C strings containing the subfile names. Memory is allocated by the function and must be freed by the caller.   
[out] | len | Pointer to the number of subfiles in the `filenames` array. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5FDsubfiling_get_file_mapping()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7 "Retrieve the subfile names for an HDF5 file using the subfiling VFD.") retrieves the names of all physical subfiles that collectively make up a logical HDF5 file using the subfiling Virtual File Driver (VFD). The subfiling VFD distributes file data across multiple subfiles to improve parallel I/O performance, particularly systems where metadata operations can become a bottleneck.
The function returns an array of subfile names corresponding to the physical files stored on the file system. Each MPI rank may be responsible for different subfiles, and this function provides visibility into which subfiles are associated with the calling rank.
**Typical use cases include:**
  * **File management** : Understanding the physical storage layout for backup and archival
  * **Performance analysis** : Identifying subfile distribution patterns
  * **File fusion** : Providing subfile lists to tools like h5fuse for recombining files
  * **Debugging** : Troubleshooting subfiling configuration and I/O patterns
  * **Storage optimization** : Analyzing subfile sizes and distribution



Note
     **Memory management** : The caller must call [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") on each string in the `filenames` array, and then call [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") on the `filenames` array itself: 
for (size_t i = 0; i < len; i++) {
[H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)(filenames[i]);
}
[H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)(filenames);
[H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)
herr_t H5free_memory(void *mem)
Frees memory allocated by the HDF5 library. 

Note
     **VFD requirement** : This function only works with files that use the subfiling VFD. Calling it on files using other VFDs will result in an error. This function will not be accessible if support for the subfiling VFD is unavailable or disabled.  

Note
     **MPI context** : The function returns subfiles associated with the calling MPI rank. Different ranks may receive different subfile lists depending on the subfiling configuration and I/O concentrator (IOC) assignment.  

Note


Warning


Example
    
#include "hdf5.h"
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id;
char **subfile_names = NULL;
size_t num_subfiles;
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) ret;
// Open file with subfiling VFD (setup not shown)
// ...
// Get subfile mapping
ret = [H5FDsubfiling_get_file_mapping](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7)(file_id, &subfile_names, &num_subfiles);
if (ret >= 0) {
printf("Found %zu subfiles:\n", num_subfiles);
for (size_t i = 0; i < num_subfiles; i++) {
printf(" %s\n", subfile_names[i]);
[H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)(subfile_names[i]); // Free each string
}
[H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)(subfile_names); // Free the array
} else {
printf("Error getting file mapping\n");
}
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5FDsubfiling_get_file_mapping](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga01c9bf33d6c925f006684acfa9b9dff7)
H5_DLL herr_t H5FDsubfiling_get_file_mapping(hid_t file_id, char ***filenames, size_t *len)
Retrieve the subfile names for an HDF5 file using the subfiling VFD. 

See also
     [H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.")      [H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.")      [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") 

Since
    2.0.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


