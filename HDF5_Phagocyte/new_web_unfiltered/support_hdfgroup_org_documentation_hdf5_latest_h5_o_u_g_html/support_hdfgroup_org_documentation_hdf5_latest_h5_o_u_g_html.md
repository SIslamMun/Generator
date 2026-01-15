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
  * [ The HDF5 Object Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title1 "H2")
  * [ Object Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title2 "H2")
  * [ Object Traversal](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title3 "H2")
  * [ Object Comments](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title4 "H2")
  * [ Reference Counting](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title5 "H2")
  * [ Cache Control](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title6 "H2")
  * [ Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_o__u_g.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Objects
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Object Interface
##  Introduction
The HDF5 Object interface (H5O) provides functions for managing HDF5 objects at a generic level. While specific object types (groups, datasets, datatypes) have their own interfaces ([Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html), [Datasets (H5D)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html), [Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)), the H5O interface enables operations that apply to all object types uniformly. 

See also
     [Objects (H5O)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html) Reference Manual
##  Object Operations
The H5O interface provides several categories of operations:
  * **Object Information** : Retrieve metadata about objects using [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5), [H5Oget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c), [H5Oget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd), and [H5Oget_native_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46 "Retrieve native file format information about an object."). These functions return information such as reference counts, modification times, and storage details.


  * **Object Copying** : Copy objects between locations or files using [H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file."). This creates a complete copy of the source object including all its attributes and, for groups, can optionally copy contained objects recursively.


  * **Object Opening** : Open objects by address or index using [H5Oopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9f635f58c7ddf17f87c253bfbca08bc1 "Opens an object in an HDF5 file by location identifier and path name."), [H5Oopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb66e5cbb3ca79890fc284a0b06762be "Opens the nth object in a group."), or [H5Oopen_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262 "Opens an object in an HDF5 file using its VOL independent token."). This enables access to objects when the specific type is unknown.


  * **Object Linking** : Create links to objects using [H5Olink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2c97dd58e64b67d16325fceb7e02113f "Creates a hard link to an object in an HDF5 file."), enabling multiple names to reference the same object.


##  Object Traversal
The H5O interface provides powerful traversal capabilities:
  * [H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) recursively visits all objects in a group hierarchy 
  * [H5Ovisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gab02a69e88b11404e7fd61f55344b186c) performs recursive visitation starting from a named location


These functions invoke a user-supplied callback for each object encountered, enabling custom processing, cataloging, or analysis of file contents.
##  Object Comments
Objects can have associated comment strings for documentation purposes:
  * [H5Oset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga8b5cf8e916204e29616516046121f631 "Sets comment for specified object.") and [H5Oset_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafeb5242de7f1080b5c19f4fe19784505 "Sets comment for specified object.") attach comments to objects 
  * [H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b "Retrieves comment for specified object.") and [H5Oget_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gae6d92d597c5a292d342a1bda91e41171 "Retrieves comment for specified object.") retrieve object comments


##  Reference Counting
HDF5 uses reference counting to manage object lifetimes. The library automatically deletes objects when their reference count reaches zero:
  * [H5Oincr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2086bad6c3cd2a711c306a48c093ff55 "Increments an object reference count.") manually increments an object's reference count 
  * [H5Odecr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga60c20da5e244c28a653d4fa23d316b44 "Decrements an object reference count.") manually decrements an object's reference count



Attention
    Manual manipulation of reference counts should be used with extreme caution. Improper use can lead to memory leaks or premature object deletion.
##  Cache Control
For performance optimization, the H5O interface provides cache management functions:
  * [H5Orefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0318b68be9ab23a92b8a6bee0af9e2f "Refreshes all buffers associated with an HDF5 object.") reloads object metadata from disk, discarding cached data 
  * [H5Oflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gad99f35048cba4534b6393214684f090f "Flushes all buffers associated with an HDF5 object to disk.") writes object metadata to disk immediately


These functions are particularly useful in SWMR (Single Writer Multiple Reader) scenarios.
##  Summary
The H5O interface provides essential generic object operations: 
  * Query object metadata and properties across all object types 
  * Copy objects within and between HDF5 files 
  * Traverse object hierarchies with custom processing 
  * Manage object comments for documentation 
  * Control object caching and persistence


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


