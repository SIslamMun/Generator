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
  * [ h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title2 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title3 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title4 "H2")
  * [ Usage Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5repack Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5repack
##  Introduction
With h5repack, you can write an HDF5 file to a new file and change the layout for objects in the new file.
##  Usage
#### h5repack [OPTIONS] file1 file2
  * **file1** [Input](https://support.hdfgroup.org/documentation/hdf5/latest/struct_input.html) HDF5 File 
  * **file2** Output HDF5 File


##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–verbose=N** Verbose mode, print object information. N - is an integer greater than 1, 2 displays read/write timing 
  * **–version** Print the library version number and exit 
  * **–native** Use a native HDF5 type when repacking 
  * **–page-buffer-size=N** Set the page buffer cache size, N=non-negative integers 
  * **–src-vol-value** Value (ID) of the VOL connector to use for opening the input HDF5 file specified 
  * **–src-vol-name** Name of the VOL connector to use for opening the input HDF5 file specified 
  * **–src-vol-info** VOL-specific info to pass to the VOL connector used for opening the input HDF5 file specified 
  * **–dst-vol-value** Value (ID) of the VOL connector to use for opening the output HDF5 file specified 
  * **–dst-vol-name** Name of the VOL connector to use for opening the output HDF5 file specified 
  * **–dst-vol-info** VOL-specific info to pass to the VOL connector used for opening the output HDF5 file specified 
  * **–src-vfd-value** Value (ID) of the VFL driver to use for opening the input HDF5 file specified 
  * **–src-vfd-name** Name of the VFL driver to use for opening the input HDF5 file specified 
  * **–src-vfd-info** VFD-specific info to pass to the VFL driver used for opening the input HDF5 file specified 
  * **–dst-vfd-value** Value (ID) of the VFL driver to use for opening the output HDF5 file specified 
  * **–dst-vfd-name** Name of the VFL driver to use for opening the output HDF5 file specified 
  * **–dst-vfd-info** VFD-specific info to pass to the VFL driver used for opening the output HDF5 file specified 
  * **–latest** Use latest version of file format This option will take precedence over the options –low and –high 
  * **–low=BOUND** The low bound for library release versions to use when creating objects in the file (default is [H5F_LIBVER_EARLIEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1)) 
  * **–high=BOUND** The high bound for library release versions to use when creating objects in the file (default is [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)) 
  * **–merge** Follow external soft link recursively and merge data 
  * **–prune** Do not follow external soft links and remove link 
  * **–merge –prune** Follow external link, merge data and remove dangling link 
  * **–compact=L1** Maximum number of links in header messages 
  * **–indexed=L2** Minimum number of links in the indexed format 
  * **–ssize=S[:F]** Shared object header message minimum size 
  * **–minimum=M** Do not apply the filter to datasets smaller than M 
  * **–file=E** Name of file E with the –file and –layout options 
  * **–ublock=U** Name of file U with user block data to be added 
  * **–block=B** Size of user block to be added 
  * **–metadata_block_size=A** Metadata block size for [H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.")
  * **–threshold=T** Threshold value for [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.")
  * **–alignment=A** Alignment value for [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.")
  * **–sort_by=Q** Sort groups and attributes by index Q 
  * **–sort_order=Z** Sort groups and attributes by order Z 
  * **–filter=FILT** Filter type 
  * **–layout=LAYT** Layout type 
  * **–fs_strategy=FS_STRATEGY** File space management strategy for [H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...")
  * **–fs_persist=FS_PERSIST** Persisting or not persisting free-space for [H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...")
  * **–fs_threshold=FS_THRESHOLD** : Free-space section threshold for [H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...")
  * **–fs_pagesize=FS_PAGESIZE** File space page size for [H5Pset_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963 "Sets the file space page size for a file creation property list.")


###  Arguments to Certain Options
  * **M** - is an integer greater than 1, size of dataset in bytes (default is 0) 
  * **E** - is a filename. 
  * **S** - is an integer 
  * **U** - is a filename. 
  * **T** - is an integer 
  * **A** - is an integer greater than zero 
  * **Q** - is the sort index type for the input file. It can be "name" or "creation_order" (default) 
  * **Z** - is the sort order type for the input file. It can be "descending" or "ascending" (default) 
  * **B** - is the user block size, any value that is 512 or greater and is a power of 2 (1024 default) 
  * **F** - is the shared object header message type, any of <dspace|dtype|fill| pline|attr>. If F is not specified, S applies to all messages


###  Library Version Bounds
**BOUND** is an integer indicating the library release versions to use when creating objects in the file (see [H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.")): 
  * **0** This is [H5F_LIBVER_EARLIEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * **1** This is [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * **2** This is [H5F_LIBVER_V110](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a33c6cdc401a3a32dbf63d74019fad4b3) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * **3** This is [H5F_LIBVER_V112](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2ac1b80c0874a79934aef1f7f44863853b) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * **4** This is [H5F_LIBVER_V114](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a9ecfa96a6f297e218c21c715312d74de) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * **5** This is [H5F_LIBVER_V200](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a7e8e06822de9c57c31c7dd404520974d) in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct 
  * [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0) is aliased to [H5F_LIBVER_V200](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a7e8e06822de9c57c31c7dd404520974d) for this release


###  File Strategy Settings
**FS_STRATEGY** is a string indicating the file space strategy used: 
  * **FSM_AGGR** The mechanisms used in managing file space are free-space managers, aggregators and virtual file driver. 
  * **PAGE** The mechanisms used in managing file space are free-space managers with embedded paged aggregation and virtual file driver. 
  * **AGGR** The mechanisms used in managing file space are aggregators and virtual file driver. 
  * **NONE** The mechanisms used in managing file space are virtual file driver. 
  * The default strategy when not set is **FSM_AGGR** without persisting free-space.


  * **FS_PERSIST** is 1 for persisting free-space or 0 for not persisting free-space. The default when not set is not persisting free-space. The value is ignored for **AGGR** and **NONE** strategies.


  * **FS_THRESHOLD** is the minimum size (in bytes) of free-space sections to be tracked by the library. The default when not set is 1. The value is ignored for **AGGR** and **NONE** strategies.


  * **FS_PAGESIZE** is the size (in bytes) >=512 that is used by the library when the file space strategy **PAGE** is used. The default when not set is 4096.


###  Applying a Third-party Filter
**FILT** - is a string with the format:
  * **< objects list>:<name of filter>=<filter parameters>**


  * **< objects list>** is a comma separated list of object names, meaning apply compression only to those objects. If no names are specified, the filter is applied to all objects 
  * **< name of filter>** can be: 
    * **GZIP** to apply the HDF5 GZIP filter (GZIP compression) 
    * **SZIP** to apply the HDF5 SZIP filter (SZIP compression) 
    * **SHUF** to apply the HDF5 shuffle filter 
    * **FLET** to apply the HDF5 checksum filter 
    * **NBIT** to apply the HDF5 NBIT filter (NBIT compression) 
    * **SOFF** to apply the HDF5 Scale/Offset filter 
    * **UD** to apply a user defined filter 
    * **NONE** to remove all filters
  * **< filter parameters>** is optional filter parameter information 
    * **GZIP= <deflation level>** from 1-9 
    * **SZIP= <pixels per block,coding>** pixels per block is a even number in 2-32 and coding method is either EC or NN 
    * **SHUF** (no parameter) 
    * **FLET** (no parameter) 
    * **NBIT** (no parameter) 
    * **SOFF= <scale_factor,scale_type>** scale_factor is an integer and scale_type is either IN or DS 
    * **UD= <filter_number,filter_flag,cd_value_count,value1[,value2,...,valueN]>**
      * **Required values** filter_number, filter_flag, cd_value_count, value1 
      * **Optional values** value2 to valueN 
      * **filter_flag** 1 is OPTIONAL or 0 is MANDATORY
    * **NONE** (no parameter)


###  Layout Settings
**LAYT** - is a string with the format:
  * **< objects list>:<layout type>=<layout parameters>**


  * **< objects list>** is a comma separated list of object names, meaning that layout information is supplied for those objects. If no names are specified, the layout type is applied to all objects 
  * **< layout type>** can be: 
    * **CHUNK** to apply chunking layout 
    * **COMPA** to apply compact layout 
    * **CONTI** to apply contiguous layout
  * **< layout parameters>** is optional layout information 
    * **CHUNK=DIM[xDIM...xDIM]** , the chunk size of each dimension 
    * **COMPA** (no parameter) 
    * **CONTI** (no parameter)


###  NOTE
The environment variable **H5TOOLS_BUFSIZE** can be set to the number of MBs to change the default hyperslab buffer size from 32MB. 
setenv H5TOOLS_BUFSIZE=64 to double the hyperslab buffer. 
##  Usage Examples
  * 1) h5repack –verbose –filter=GZIP=1 file1 file2 ```
 GZIP compression with level 1 to all objects

```



  * 2) h5repack –verbose –filter=dset1:SZIP=8,NN file1 file2 ```
 SZIP compression with 8 pixels per block and NN coding method to object dset1

```



  * 3) h5repack –verbose –layout=dset1,dset2:CHUNK=20x10 –filter=dset3,dset4,dset5:NONE file1 file2 ```
 Chunked layout, with a layout size of 20x10, to objects dset1 and dset2
 and remove filters to objects dset3, dset4, dset5

```



  * 4) h5repack –latest –compact=10 –ssize=20:dtype file1 file2 ```
 Using latest file format with maximum compact group size of 10 and
 minimum shared datatype size of 20

```



  * 5) h5repack –filter=SHUF –filter=GZIP=1 file1 file2 ```
 Add both filters SHUF and GZIP in this order to all datasets

```



  * 6) h5repack –filter=UD=307,0,1,9 file1 file2 ```
 Add bzip2 filter to all datasets

```



  * 7) h5repack –low=0 –high=1 file1 file2 ```
 Set low=H5F_LIBVER_EARLIEST and high=H5F_LIBVER_V18 via
 H5Pset_libver_bounds() when creating the repacked file, file2

```



Previous Chapter [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) - Next Chapter [h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#sec_cltools_h5stat)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


