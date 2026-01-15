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
  * [ The HDF5 Link Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title1 "H2")
  * [ Link Types](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title2 "H2")
  * [ Creating Links](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title3 "H2")
  * [ Link Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title4 "H2")
  * [ Link Traversal](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title5 "H2")
  * [ User-Defined Link Types](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title6 "H2")
  * [ Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Links
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Link Interface
##  Introduction
The HDF5 Link interface (H5L) provides functions for creating, querying, and manipulating links between objects in an HDF5 file. Links are the primary mechanism for organizing objects in the HDF5 file hierarchy and enabling navigation between related data.
##  Link Types
HDF5 supports three types of links:
  * **Hard Links** : Direct references to objects within the same file. An object exists as long as at least one hard link points to it. Hard links cannot cross file boundaries.


  * **Soft Links (Symbolic Links)** : Path-based references that store the path to an object as a string. Soft links can "dangle" if the target object is deleted or moved.


  * **External Links** : References to objects in other HDF5 files. External links store both the filename and the path to the object within that file.


##  Creating Links
Links can be created using several methods:
  * [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object.") creates hard links to objects 
  * [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.") creates soft (symbolic) links using path strings 
  * [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") creates links to objects in other HDF5 files


##  Link Operations
The H5L interface provides functions for:
  * **Querying** : [H5Lexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") checks if a link exists, [H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) retrieves link information 
  * **Moving/Copying** : [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") and [H5Lcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...") relocate or duplicate links 
  * **Deleting** : [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") removes links from the file 
  * **Iterating** : [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) and [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) traverse links in groups


##  Link Traversal
The H5L interface includes powerful traversal functions that can iterate over all links in a group or recursively visit all links in a group hierarchy:
  * [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) iterates over links in a group using a group identifier 
  * [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) recursively visits all links in a group and its subgroups using a group identifier 
  * [H5Literate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga655b002428e0176c2fa23a0315fbbcc2) and [H5Lvisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga138405315e233673741893e4e250f055) provide the same functionality but accept a location identifier and group name, allowing traversal without first opening the group


These functions accept callbacks that are invoked for each link, enabling custom processing.
##  User-Defined Link Types
HDF5 allows registration of custom link types through [H5Lregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class."). User-defined links can implement custom traversal and access behaviors. Use [H5Lunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec "Unregisters a class of user-defined links.") to remove user-defined link classes and [H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2 "Determines whether a class of user-defined links is registered.") to check if a link class is registered.
##  Summary
The H5L interface provides flexible mechanisms for organizing and navigating HDF5 files: 
  * Hard links provide reference-counted object management 
  * Soft links enable flexible, path-based references 
  * External links connect objects across multiple files 
  * Iteration functions enable systematic traversal of file hierarchies 
  * User-defined links support extensibility for custom use cases


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


