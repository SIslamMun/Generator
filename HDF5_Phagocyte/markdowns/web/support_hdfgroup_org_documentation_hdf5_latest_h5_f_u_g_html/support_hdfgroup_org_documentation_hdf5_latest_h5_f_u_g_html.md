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
  * [ The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title1 "H2")
  * [ File Access Modes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title2 "H2")
  * [ File Creation and File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title3 "H2")
  * [ Low-level File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title4 "H2")
  * [ Programming Model for Files](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title5 "H2")
  * [ Using h5dump to View a File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title6 "H2")
  * [ File Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title7 "H2")
  * [ Creating or Opening an HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title8 "H2")
  * [ Closing an HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title9 "H2")
  * [ File Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title10 "H2")
  * [ Alternate File Storage Layouts and Low-level File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title11 "H2")
  * [ Code Examples for Opening and Closing Files](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title12 "H2")
  * [ Working with Multiple HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#title13 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 File
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 File
##  Introduction
The purpose of this chapter is to describe how to work with HDF5 data files.
If HDF5 data is to be written to or read from a file, the file must first be explicitly created or opened with the appropriate file driver and access privileges. Once all work with the file is complete, the file must be explicitly closed.
This chapter discusses the following: 
  * File access modes 
  * Creating, opening, and closing files 
  * The use of file creation property lists 
  * The use of file access property lists 
  * The use of low-level file drivers


This chapter assumes an understanding of the material presented in the data model chapter. For more information, see [The HDF5 Data Model and File Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html#sec_data_model).
##  File Access Modes
There are two issues regarding file access: 
  * What should happen when a new file is created but a file of the same name already exists? Should the create action fail, or should the existing file be overwritten? 
  * Is a file to be opened with read-only or read-write access?


Four access modes address these concerns. Two of these modes can be used with [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file."), and two modes can be used with [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."). 
  * [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") accepts [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90) or [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
  * [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") accepts [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) or [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)


The access modes are described in the table below.
Access flags and modes Access Flag  | Resulting Access Mode   
---|---  
[H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90) | If the file already exists, [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") fails. If the file does not exist, it is created and opened with read-write access. (Default)   
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) | If the file already exists, the file is opened with read-write access, and new data will overwrite any existing data. If the file does not exist, it is created and opened with read-write access.   
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) | An existing file is opened with read-only access. If the file does not exist, [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") fails. (Default)   
[H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) | An existing file is opened with read-write access. If the file does not exist, [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") fails.   
By default, [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") opens a file for read-only access; passing [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) allows read-write access to the file.
By default, [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") fails if the file already exists; only passing [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) allows the truncating of an existing file.
##  File Creation and File Access Properties
File creation and file access property lists control the more complex aspects of creating and accessing files.
File creation property lists control the characteristics of a file such as the size of the userblock, a user-definable data block; the size of data address parameters; properties of the B-trees that are used to manage the data in the file; and certain HDF5 Library versioning information.
For more information, see [File Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsubsec_file_property_lists_props).
This section has a more detailed discussion of file creation properties. If you have no special requirements for these file characteristics, you can simply specify [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) for the default file creation property list when a file creation property list is called for.
File access property lists control properties and means of accessing a file such as data alignment characteristics, metadata block and cache sizes, data sieve buffer size, garbage collection settings, and parallel I/O. Data alignment, metadata block and cache sizes, and data sieve buffer size are factors in improving I/O performance.
For more information, see [File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsubsec_file_property_lists_access).
This section has a more detailed discussion of file access properties. If you have no special requirements for these file access characteristics, you can simply specify [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) for the default file access property list when a file access property list is called for.
Figure 10 - More sample file structures ![](https://support.hdfgroup.org/documentation/hdf5/latest/UML_FileAndProps.gif) UML model for an HDF5 file and its property lists  
---  
##  Low-level File Drivers
The concept of an HDF5 file is actually rather abstract: the address space for what is normally thought of as an HDF5 file might correspond to any of the following at the storage level: 
  * Single file on a standard file system 
  * Multiple files on a standard file system 
  * Multiple files on a parallel file system 
  * Block of memory within an application's memory space 
  * More abstract situations such as virtual files


This HDF5 address space is generally referred to as an HDF5 file regardless of its organization at the storage level.
HDF5 accesses a file (the address space) through various types of low-level file drivers. The default HDF5 file storage layout is as an unbuffered permanent file which is a single, contiguous file on local disk. Alternative layouts are designed to suit the needs of a variety of systems, environments, and applications.
##  Programming Model for Files
Programming models for creating, opening, and closing HDF5 files are described in the sub-sections below.
###  Creating a New File
The programming model for creating a new HDF5 file can be summarized as follows: 
  * Define the file creation property list 
  * Define the file access property list 
  * Create the file


First, consider the simple case where we use the default values for the property lists. See the example below.
_Creating an HDF5 file using property list defaults_
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) ("SampleFile.h5", [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))
[H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90)
#define H5F_ACC_EXCL
**Definition** H5Fpublic.h:31
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
Note: The example above specifies that [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") should fail if SampleFile.h5 already exists.
A more complex case is shown in the example below. In this example, we define file creation and access property lists (though we do not assign any properties), specify that [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") should fail if SampleFile.h5 already exists, and create a new file named SampleFile.h5. The example does not specify a driver, so the default driver, [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606), will be used.
_Creating an HDF5 file using property lists_
fcplist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977))
<...set desired file creation properties...>
faplist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))
<...set desired file access properties...>
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) ("SampleFile.h5", [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90), fcplist_id, faplist_id)
[H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977)
#define H5P_FILE_CREATE
**Definition** H5Ppublic.h:52
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
Notes:
  1. A root group is automatically created in a file when the file is first created.
  2. File property lists, once defined, can be reused when another file is created within the same application.


###  Opening an Existing File
The programming model for opening an existing HDF5 file can be summarized as follows: 
  * Define or modify the file access property list including a low-level file driver (optional) 
  * Open the file


The code in the example below shows how to open an existing file with read-only access.
_Opening an HDF5 file_
faplist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))
status = [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052) (faplist_id)
file_id = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) ("SampleFile.h5", [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), faplist_id)
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394)
#define H5F_ACC_RDONLY
**Definition** H5Fpublic.h:28
[H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052)
herr_t H5Pset_fapl_stdio(hid_t fapl_id)
Sets the standard I/O driver.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
###  Closing a File
The programming model for closing an HDF5 file is very simple: 
  * Close file


We close SampleFile.h5 with the code in the example below.
_Closing an HDF5 file_
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) (file_id)
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
Note that [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") flushes all unwritten data to storage and that file_id is the identifier returned for SampleFile.h5 by [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.").
More comprehensive discussions regarding all of these steps are provided below.
##  Using h5dump to View a File
[h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) is a command-line utility that is included in the HDF5 distribution. This program provides a straight-forward means of inspecting the contents of an HDF5 file. You can use [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) to verify that a program is generating the intended HDF5 file. [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) displays ASCII output formatted according to the HDF5 DDL grammar.
The following [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) command will display the contents of SampleFile.h5: 
h5dump SampleFile.h5
If no datasets or groups have been created in and no data has been written to the file, the output will look something like the following: 
HDF5 "SampleFile.h5" {
GROUP "/" {
}
}
Note that the root group, indicated above by **/** , was automatically created when the file was created.
[h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) is described on the [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewContent_h5dump) page under [Command-line Tools](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_command.html). The HDF5 DDL grammar is described in the document [DDL in BNF for HDF5 2.0.0 and above](https://support.hdfgroup.org/documentation/hdf5/latest/_d_d_l_b_n_f200.html).
##  File Function Summaries
General library ([Library General (H5)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html) functions and macros), ([Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html) functions), file related ([Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) functions), and file driver ([Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) functions) are listed below.
General library functions and macros Function  | Purpose   
---|---  
[H5check_version](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.") | Verifies that HDF5 library versions are consistent.   
[H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") | Flushes all data to disk, closes all open identifiers, and cleans up memory.   
[H5dont_atexit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga7f80eb63b5e78812b9d0d50ac46764e8 "Instructs library not to install atexit\(\) cleanup routine.") | Instructs the library not to install the atexit cleanup routine.   
[H5garbage_collect](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef "Garbage collects on all free-lists of all types.") | Performs garbage collection (GC) on all free-lists of all types.   
[H5get_libversion](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaf87da966fdf896ec7bca794e21d4ab0a "Returns the HDF library release number.") | Returns the HDF library release number.   
[H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library.") | Initializes the HDF5 library.   
[H5set_free_list_limits](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545 "Sets free-list size limits.") | Sets free-list size limits.   
[H5_VERSION_GE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa80cec001d4154b554aa2b0b6223a7f3 "Determines whether the version of the library being used is greater than or equal to the specified ve...") | Determines whether the version of the library being used is greater than or equal to the specified version.   
[H5_VERSION_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#acc311aa4fed2832a44c4bb93f2625043 "Determines whether the version of the library being used is less than or equal to the specified versi...") | Determines whether the version of the library being used is less than or equal to the specified version.   
File functions  Function  | Purpose   
---|---  
[H5Fclear_elink_file_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gafcc153d8606829d4401e93305e5246d7 "Clears the external link open file cache.") | Clears the external link open file cache for a file.   
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") | Closes HDF5 file.   
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") | Creates new HDF5 file.   
[H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.") | Flushes data to HDF5 file on storage medium.   
[H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4 "Returns a file access property list identifier.") | Returns a file access property list identifier.   
[H5Fget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga2f823a9e929b00b06a6be80619a61778 "Returns a file creation property list identifier.") | Returns a file creation property list identifier.   
[H5Fget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file.") | Retrieves a copy of the image of an existing, open file.   
[H5Fget_filesize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga515426821321c261a825b4e4a3f576fe "Returns the size of an HDF5 file \(in bytes\)") | Returns the size of an HDF5 file.   
[H5Fget_freespace](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3ef2673183567543346668a8f1eca2e9 "Returns the amount of free space in a file \(in bytes\)") | Returns the amount of free space in a file.   
[H5Fget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae17036b3e36a8777328204e8bf073144) | Returns global information for a file.   
[H5Fget_intent](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga466179d7783d256329c2e3110055a16c "Determines the read/write or read-only status of a file.") | Determines the read/write or read-only status of a file.   
[H5Fget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59 "Obtains current metadata cache configuration for target file.") | Obtains current metadata cache configuration for target file.   
[H5Fget_mdc_hit_rate](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f "Obtains target file's metadata cache hit rate.") | Obtains target file's metadata cache hit rate.   
[H5Fget_mdc_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f "Obtains current metadata cache size data for specified file.") | Obtains current metadata cache size data for specified file.   
[H5Fget_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd "Retrieves the atomicity mode in use.") | Retrieves the atomicity mode in use.   
[H5Fget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga0ed43dbe476a160b73f55127c7db797c "Retrieves name of file to which object belongs.") | Retrieves the name of the file to which the object belongs.   
[H5Fget_obj_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadcdae0aca7c88064db0d32de7f1e31f2 "Returns the number of open object identifiers for an open file.") | Returns the number of open object identifiers for an open file.   
[H5Fget_obj_ids](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga35e72579bd07433162b80ddc0bd0c5b1 "Returns a list of open object identifiers.") | Returns a list of open object identifiers.   
[H5Fget_vfd_handle](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") | Returns pointer to the file handle from the virtual file driver.   
[H5Fis_hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225 "Determines whether a file is in the HDF5 format.") | Determines whether a file is in the HDF5 format.   
[H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") | Mounts a file.   
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") | Opens an existing HDF5 file.   
[H5Freopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3f213eb05c5419d63ba168c30036e47b "Returns a new identifier for a previously-opened HDF5 file.") | Returns a new identifier for a previously-opened HDF5 file.   
[H5Freset_mdc_hit_rate_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f "Resets hit rate statistics counters for the target file.") | Resets hit rate statistics counters for the target file.   
[H5Fset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8 "Attempts to configure metadata cache of target file.") | Configures metadata cache of target file.   
[H5Fset_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") | Sets the MPI atomicity mode.   
[H5Funmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga79e80cbf6d9782402b353b763162779c "Un-mounts an HDF5 file.") | Unmounts a file.   
File creation property list functions (H5P) 
File creation property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d "Sets user block size.")/[H5Pget_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga75b312bb0c70419fc428d743a65bed86 "Retrieves the size of a user block.") | Sets/retrieves size of userblock.   
[H5Pset_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae5eb3ba16f063d151d1b56d33e0710a9 "Sets the byte size of the offsets and lengths used to address objects in an HDF5 file.")/[H5Pget_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga8da25b0367cf226c2888141661fd7a2d "Retrieves the size of the offsets and lengths used in an HDF5 file.") | Sets/retrieves byte size of offsets and lengths used to address objects in HDF5 file.   
[H5Pset_sym_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga444ca905f084f9f96b7fe60d2a8c8176)/[H5Pget_sym_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga1d4ee26c030ced6d7a314543578c88b1 "Retrieves the size of the symbol table B-tree 1/2 rank and the symbol table leaf node 1/2 size.") | Sets/retrieves size of parameters used to control symbol table nodes.   
[H5Pset_istore_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga84a72f59d17841c37ab34674bf22a10c "Sets the size of the parameter used to control the B-trees for indexing chunked datasets.")/[H5Pget_istore_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga2179b032be5d2efbca63d8f82a292ec1 "Queries the 1/2 rank of an indexed storage B-tree.") | Sets/retrieves size of parameter used to control B-trees for indexing chunked datasets.   
[H5Pset_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963 "Sets the file space page size for a file creation property list.")/[H5Pget_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gaab5e8c08e4f588e0af1d937fcebfc885 "Retrieves the file space page size for a file creation property list.") | Sets or retrieves the file space page size used in paged aggregation and paged buffering.   
[H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...")/[H5Pget_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae43e976d78a41b3b181fa10f8c77f522 "Retrieves the file space handling strategy, persisting free-space condition and threshold value for a...") | Sets or retrieves the file space handling strategy, the persisting free-space and the free-space section size.   
[H5Pset_shared_mesg_nindexes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga5e5020b1e2579da4617ea115e3cc50f1 "Sets number of shared object header message indexes.")/[H5Pget_shared_mesg_nindexes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga30980db1814a251e7b40362af1006652 "Retrieves the number of shared object header message indexes in file creation property list.") | Sets or retrieves number of shared object header message indexes in file creation property list.   
[H5Pset_shared_mesg_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga052aba0c1c5a3908a62335fc28e287ef "Configures the specified shared object header message index.") | Configures the specified shared object header message index.   
[H5Pget_shared_mesg_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gac6bac4446c45d348c953b3afdecede2c "Retrieves the configuration settings for a shared message index.") | Retrieves the configuration settings for a shared message index.   
[H5Pset_shared_mesg_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga967f961f4002d63804dc67b3bcd8f354 "Sets shared object header message storage phase change thresholds.")/[H5Pget_shared_mesg_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gab013e791706b44f545a97096d8e4c72e "Retrieves shared object header message phase change information.") | Sets or retrieves shared object header message storage phase change thresholds.   
[H5Pget_version](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga99c0afbb68e8e775ae70cac44404a534 "Retrieves the version information of various objects for a file creation property list\(deprecated\)") |   
File access property list functions (H5P) 
File access property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.")/[H5Pget_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6735afde382cfd746b92a1a3b0e6a2ab "Retrieves the current settings for alignment properties from a file access property list.") | Sets/retrieves alignment properties.   
[H5Pset_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.")/[H5Pget_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861 "Queries the raw data chunk cache parameters.") | Sets/retrieves metadata cache and raw data chunk cache parameters.   
[H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250 "Sets write tracking information for core driver, H5FD_CORE.")/[H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469 "Gets information about the write tracking feature used by the core VFD.") | Sets/retrieves write tracking information for core driver.   
[H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.")/[H5Pget_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923 "Retrieves the size of the external link open file cache.") | Sets/retrieves the size of the external link open file cache from the specified file access property list.   
[H5Pset_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaca1f462af9b441a0fdc6d0da6d4e17de "Controls the library's behavior of evicting metadata associated with a closed object.")/[H5Pget_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga465a52e20e031a604b719bbfe9883b66 "Retrieves the file access property list setting that determines whether an HDF5 object will be evicte...") | Set/get the file access property list setting that determines whether an HDF5 object will be evicted from the library's metadata cache when it is closed.   
[H5Pset_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f "Sets garbage collecting references flag.")/[H5Pget_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa81d8427b419d80eff6e1d216d99b71 "Returns garbage collecting references setting.") | Sets/retrieves garbage collecting references flag.   
[H5Pset_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6b24e6daf4816bbfb89b63bab40aa982 "Sets offset property for low-level access to a file in a family of files.") | Sets offset property for low-level access to a file in a family of files.   
[H5Pget_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14977eaaf6565ba871b575de3163f1b3 "Retrieves a data offset from the file access property list.") | Retrieves a data offset from the file access property list.   
[H5Pset_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.")/[H5Pget_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga41da04bb4f823ba9f7d6c57dc8fe2878 "Returns the file close degree.") | Sets/retrieves file close degree property.   
[H5Pset_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") | Sets an initial file image in a memory buffer.   
[H5Pget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file.") | Retrieves a copy of the file image designated as the initial content and structure of a file.   
[H5Pset_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.")/[H5Pget_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images.") | Sets/gets the callbacks for working with file images.   
[H5Pset_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaae4de80c533a38da9f269c02b8e71de2 "Sets the file locking property values.")/[H5Pget_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6f68d03de151117cefa29b571865a394 "Retrieves the file locking property values.") | Sets/retrieves file locking property values.   
[H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.")/[H5Pget_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac17861181246af0209c0da5209305461 "Returns the current metadata block size setting.") | Sets the minimum metadata blocksize or retrieves the current metadata block size setting.   
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list.")/[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e "Retrieves the number of read attempts from a file access property list.") | Sets/gets the number of read attempts from a file access property list.   
[H5Pset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3 "Set the initial metadata cache configuration in the indicated File Access Property List to the suppli...")/[H5Pget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee "Get the current initial metadata cache configuration from the provided file access property list.") | Set/get the initial metadata cache configuration in the indicated file access property list.   
[H5Pset_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434 "Sets the metadata cache image option for a file access property list.")/[H5Pget_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa18d59ee9efb12626410b1638f76f00 "Retrieves the metadata cache image configuration values for a file access property list.") | Set/get the metadata cache image option for a file access property list.   
[H5Pset_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.")/[H5Pget_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") | Set/get the metadata cache logging options.   
[H5Pset_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.")/[H5Pget_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe "Retrieves type of data property for MULTI driver.") | Sets/gets the type of data property for the MULTI driver.   
[H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19 "Sets a callback function to invoke when an object flush occurs in the file.")/[H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e "Retrieves the object flush property values from the file access property list.") | Set/get the object flush property values from the file access property list.   
[H5Pset_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8008cddafa81bd1ddada23f6d9a161ca "Sets the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.")/[H5Pget_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0da11baf31cf424d053aa7952c933d98 "Retrieves the maximum size for the page buffer and the minimum percentage for metadata and raw data p...") | Set/get the maximum size for the page buffer.   
[H5Pset_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee "Sets the maximum size of the data sieve buffer.")/[H5Pget_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66 "Returns maximum data sieve buffer size.") | Sets/retrieves maximum size of data sieve buffer.   
[H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") | Sets bounds on library versions, and indirectly format versions, to be used when creating objects.   
[H5Pget_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad5d7e671c3a06bcee64bc25841aaf607 "Retrieves library version bounds settings that indirectly control the format versions used when creat...") | Retrieves library version bounds settings that indirectly control the format versions used when creating objects.   
[H5Pset_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5a99962a79412814b79be830f14c23dd "Sets the size of a contiguous block reserved for small data.") | Sets the size of a contiguous block reserved for small data.   
[H5Pget_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6896bea06d7744b56e22347f572f5470 "Retrieves the current small data block size setting.") | Retrieves the current small data block size setting.   
[H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0 "Set the file VOL connector for a file access property list.") | Sets the file VOL connector for a file access property list.   
[H5Pget_vol_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2ad4dc5c6e5e4271334a7b1c6ee0777d "Query the capability flags for the VOL connector that will be used with this file access property lis...") | Retrieves the capability flags for the VOL connector that will be used with a file access property list.   
[H5Pget_vol_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838 "Returns the identifier of the current VOL connector.") | Retrieves the identifier of the current VOL connector.   
[H5Pget_vol_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f "Returns a copy of the VOL information for a connector.") | Retrieves a copy of the VOL information for a connector.   
[H5Pset_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0 "Set the MPI communicator and info.")/[H5Pget_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5554cf0775f9d7ac3b0cd844533d4486 "Get the MPI communicator and info.") | Sets/retrieves the MPI communicator and info.   
[H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)")/[H5Pget_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.") | Sets/retrieves metadata write mode setting.   
File driver property list functions (H5P) 
File driver property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") | Sets a file driver.   
[H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") | Returns the identifier for the driver used to create a file.   
[H5Pget_driver_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56 "Returns a pointer to file driver information.") | Returns a pointer to file driver information.   
[H5Pset_driver_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga156702db27ece40d21b37be5fe5e8b15 "Sets a file driver according to a given driver name.") | Sets a file driver according to a given driver name.   
[H5Pset_driver_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac4426b1d36aa8766fbe2deaf67a18c06 "Sets a file driver according to a given driver value \(ID\).") | Sets a file driver according to a given driver value.   
[H5Pget_driver_config_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac0d9eb573b84ce125433e81b2a475085 "Retrieves a string representation of the configuration for the driver set on the given FAPL....") | Retrieves a string representation of the configuration for the driver.   
[H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.")/[H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50 "Queries core file driver properties.") | Sets the driver for buffered memory files (in RAM) or retrieves information regarding the driver.   
[H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.")/[H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d "Retrieves direct I/O driver settings.") | Sets up use of the direct I/O driver or retrieves the direct I/O driver settings.   
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.")/[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375 "Returns file access property list information.") | Sets driver for file families, designed for systems that do not support files larger than 2 gigabytes, or retrieves information regarding driver.   
[H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.")/[H5Pget_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35 "Queries a File Access Property List for H5FD_HDFS file driver properties.") | .   
[H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.")/[H5Pget_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98 "Queries a File Access Property List for H5FD_IOC file driver properties.") | Modifies/queries the file driver properties of the I/O concentrator driver.   
[H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.") | The logging driver is a clone of the standard SEC2 ([H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)) driver with additional facilities for logging metrics and activity to a file.   
[H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.")/[H5Pget_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1 "Queries a File Access Property List for H5FD_MIRROR file driver properties.") | Modifies/queries the file driver properties of the mirror driver.   
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")/[H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") | Sets driver for files on parallel file systems (MPI I/O) or retrieves information regarding the driver.   
H5Pset_fapl_mpiposix/H5Pget_fapl_mpiposix  | No longer available.   
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.")/[H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d "Returns information about the multi-file access property list.") | Sets driver for multiple files, separating categories of metadata and raw data, or retrieves information regarding driver.   
[H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list")/[H5Pget_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107 "get the onion info from the file access property list") | Modifies/queries the file driver properties of the onion driver.   
[H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.")/[H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties.") | Modifies/queries the file driver properties of the ros3 driver.   
[H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.") | Sets driver for unbuffered permanent files or retrieves information regarding driver.   
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.") | Sets driver for split files, a limited case of multi driver with one metadata file and one raw data file.   
[H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.")/[H5Pget_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6e850a0319dc6839e2dd06a0522b3831 "Gets splitter driver properties from the the file access property list.") | Modifies/queries the file driver properties of the splitter driver.   
[H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.") | Sets driver for buffered permanent files.   
[H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.")/[H5Pget_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.") | Modifies/queries the file driver properties of the subfiling driver.   
[H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.") | Sets the Windows I/O driver.   
[H5Pset_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.") | Specifies type of data to be accessed via the MULTI driver enabling more direct access.   
[H5Pget_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe "Retrieves type of data property for MULTI driver.") | Retrieves type of data property for MULTI driver.   
##  Creating or Opening an HDF5 File
This section describes in more detail how to create and how to open files.
New HDF5 files are created and opened with [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file."); existing files are opened with [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."). Both functions return an object identifier which must eventually be released by calling [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.").
To create a new file, call [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file."): 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) (const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") creates a new file named name in the current directory. The file is opened with read and write access; if the [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) flag is set, any pre-existing file of the same name in the same directory is truncated. If [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) is not set or [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90) is set and if a file of the same name exists, [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") will fail.
The new file is created with the properties specified in the property lists fcpl_id and fapl_id. fcpl is short for file creation property list. fapl is short for file access property list. Specifying [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) for either the creation or access property list will use the library's default creation or access properties.
If [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") successfully creates the file, it returns a file identifier for the new file. This identifier will be used by the application any time an object identifier, an OID, for the file is required. Once the application has finished working with a file, the identifier should be released and the file closed with [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.").
To open an existing file, call [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."): 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) (const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") opens an existing file with read-write access if [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) is set and read-only access if [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) is set.
fapl_id is the file access property list identifier. Alternatively, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) indicates that the application relies on the default I/O access parameters. Creating and changing access property lists is documented further below.
A file can be opened more than once via multiple [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") calls. Each such call returns a unique file identifier and the file can be accessed through any of these file identifiers as long as they remain valid. Each of these file identifiers must be released by calling [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") when it is no longer needed.
For more information, see [File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsubsec_file_property_lists_access).
For more information, see [File Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_property_lists).
##  Closing an HDF5 File
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") both closes a file and releases the file identifier returned by [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") or [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file."). [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") must be called when an application is done working with a file; while the HDF5 Library makes every effort to maintain file integrity, failure to call [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") may result in the file being abandoned in an incomplete or corrupted state.
To close a file, call [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."): 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
This function releases resources associated with an open file. After closing a file, the file identifier, file_id, cannot be used again as it will be undefined.
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") fulfills three purposes: to ensure that the file is left in an uncorrupted state, to ensure that all data has been written to the file, and to release resources. Use [H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.") if you wish to ensure that all data has been written to the file but it is premature to close it.
Note regarding serial mode behavior: When [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") is called in serial mode, it closes the file and terminates new access to it, but it does not terminate access to objects that remain individually open within the file. That is, if [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") is called for a file but one or more objects within the file remain open, those objects will remain accessible until they are individually closed. To illustrate, assume that a file, fileA, contains a dataset, data_setA, and that both are open when [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") is called for fileA. data_setA will remain open and accessible, including writable, until it is explicitly closed. The file will be automatically and finally closed once all objects within it have been closed.
Note regarding parallel mode behavior: Once [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") has been called in parallel mode, access is no longer available to any object within the file.
##  File Property Lists
Additional information regarding file structure and access are passed to [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") and [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") through property list objects. Property lists provide a portable and extensible method of modifying file properties via simple API functions. There are two kinds of file-related property lists: 
  * File creation property lists 
  * File access property lists


In the following sub-sections, we discuss only one file creation property, userblock size, in detail as a model for the user. Other file creation and file access properties are mentioned and defined briefly, but the model is not expanded for each; complete syntax, parameter, and usage information for every property list function is provided in the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) section of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
For more information,  

See also
     [Properties and Property Lists in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#sec_plist).
###  Creating a Property List
If you do not wish to rely on the default file creation and access properties, you must first create a property list with [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class."). 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cls_id)
cls_id is the type of property list being created. In this case, the appropriate values are [H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977) for a file creation property list and [H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e) for a file access property list.
Thus, the following calls create a file creation property list and a file access property list with identifiers fcpl_id and fapl_id, respectively: 
fcpl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977))
fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))
Once the property lists have been created, the properties themselves can be modified via the functions described in the following sub-sections.
###  File Creation Properties
File creation property lists control the file metadata, which is maintained in the superblock of the file. These properties are used only when a file is first created.
#### Userblock Size
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga75b312bb0c70419fc428d743a65bed86) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d)
herr_t H5Pset_userblock(hid_t plist_id, hsize_t size)
Sets user block size.
[H5Pget_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga75b312bb0c70419fc428d743a65bed86)
herr_t H5Pget_userblock(hid_t plist_id, hsize_t *size)
Retrieves the size of a user block.
The userblock is a fixed-length block of data located at the beginning of the file and is ignored by the HDF5 library. This block is specifically set aside for any data or information that developers determine to be useful to their applications but that will not be used by the HDF5 library. The size of the userblock is defined in bytes and may be set to any power of two with a minimum size of 512 bytes. In other words, userblocks might be 512, 1024, or 2048 bytes in size.
This property is set with [H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d "Sets user block size.") and queried via [H5Pget_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga75b312bb0c70419fc428d743a65bed86 "Retrieves the size of a user block."). For example, if an application needed a 4K userblock, then the following function call could be used: 
status = [H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d)(fcpl_id, 4096)
The property list could later be queried with: 
status = [H5Pget_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga75b312bb0c70419fc428d743a65bed86)(fcpl_id, size)
and the value 4096 would be returned in the parameter size.
Other properties, described below, are set and queried in exactly the same manner. Syntax and usage are detailed in the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) section of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
#### Offset and Length Sizes
This property specifies the number of bytes used to store the offset and length of objects in the HDF5 file. Values of 2, 4, and 8 bytes are currently supported to accommodate 16-bit, 32-bit, and 64-bit file address spaces.
These properties are set and queried via [H5Pset_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae5eb3ba16f063d151d1b56d33e0710a9 "Sets the byte size of the offsets and lengths used to address objects in an HDF5 file.") and [H5Pget_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga8da25b0367cf226c2888141661fd7a2d "Retrieves the size of the offsets and lengths used in an HDF5 file.").
#### Symbol Table Parameters
The size of symbol table B-trees can be controlled by setting the 1/2-rank and 1/2-node size parameters of the B-tree.
These properties are set and queried via [H5Pset_sym_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga444ca905f084f9f96b7fe60d2a8c8176) and [H5Pget_sym_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga1d4ee26c030ced6d7a314543578c88b1 "Retrieves the size of the symbol table B-tree 1/2 rank and the symbol table leaf node 1/2 size.")
#### Indexed Storage Parameters
The size of indexed storage B-trees can be controlled by setting the 1/2-rank and 1/2-node size parameters of the B-tree.
These properties are set and queried via [H5Pset_istore_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga84a72f59d17841c37ab34674bf22a10c "Sets the size of the parameter used to control the B-trees for indexing chunked datasets.") and [H5Pget_istore_k](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga2179b032be5d2efbca63d8f82a292ec1 "Queries the 1/2 rank of an indexed storage B-tree.").
#### Version Information
Various objects in an HDF5 file may over time appear in different versions. The HDF5 Library keeps track of the version of each object in the file.
Version information is retrieved via [H5Pget_version](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga99c0afbb68e8e775ae70cac44404a534 "Retrieves the version information of various objects for a file creation property list\(deprecated\)").
###  File Access Properties
This section discusses file access properties that are not related to the low-level file drivers. File drivers are discussed separately later in this chapter. For more information,  

See also
     [Alternate File Storage Layouts and Low-level File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_alternate_drivers).
File access property lists control various aspects of file I/O and structure.
#### Data Alignment
Sometimes file access is faster if certain data elements are aligned in a specific manner. This can be controlled by setting alignment properties via the [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") function. There are two values involved: 
  * A threshold value 
  * An alignment interval


Any allocation request at least as large as the threshold will be aligned on an address that is a multiple of the alignment interval.
#### Metadata Block Allocation Size
Metadata typically exists as very small chunks of data; storing metadata elements in a file without blocking them can result in hundreds or thousands of very small data elements in the file. This can result in a highly fragmented file and seriously impede I/O. By blocking metadata elements, these small elements can be grouped in larger sets, thus alleviating both problems.
[H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.") sets the minimum size in bytes of metadata block allocations. [H5Pget_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac17861181246af0209c0da5209305461 "Returns the current metadata block size setting.") retrieves the current minimum metadata block allocation size.
#### Metadata Cache
Metadata and raw data I/O speed are often governed by the size and frequency of disk reads and writes. In many cases, the speed can be substantially improved by the use of an appropriate cache.
[H5Pset_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") sets the minimum cache size for both metadata and raw data and a preemption value for raw data chunks. [H5Pget_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861 "Queries the raw data chunk cache parameters.") retrieves the current values.
#### Data Sieve Buffer Size
Data sieve buffering is used by certain file drivers to speed data I/O and is most commonly when working with dataset hyperslabs. For example, using a buffer large enough to hold several pieces of a dataset as it is read in for hyperslab selections will boost performance noticeably.
[H5Pset_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee "Sets the maximum size of the data sieve buffer.") sets the maximum size in bytes of the data sieve buffer. [H5Pget_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66 "Returns maximum data sieve buffer size.") retrieves the current maximum size of the data sieve buffer.
#### Garbage Collection References
Dataset region references and other reference types use space in an HDF5 file's global heap. If garbage collection is on (1) and the user passes in an uninitialized value in a reference structure, the heap might become corrupted. When garbage collection is off (0), however, and the user reuses a reference, the previous heap block will be orphaned and not returned to the free heap space. When garbage collection is on, the user must initialize the reference structures to 0 or risk heap corruption.
[H5Pset_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f "Sets garbage collecting references flag.") sets the garbage collecting references flag.
##  Alternate File Storage Layouts and Low-level File Drivers
The concept of an HDF5 file is actually rather abstract: the address space for what is normally thought of as an HDF5 file might correspond to any of the following: 
  * Single file on standard file system 
  * Multiple files on standard file system 
  * Multiple files on parallel file system 
  * Block of memory within application's memory space 
  * More abstract situations such as virtual files


This HDF5 address space is generally referred to as an HDF5 file regardless of its organization at the storage level.
HDF5 employs an extremely flexible mechanism called the virtual file layer, or VFL, for file I/O. A full understanding of the VFL is only necessary if you plan to write your own drivers see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html) in the HDF5 [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html).
For our purposes here, it is sufficient to know that the low-level drivers used for file I/O reside in the VFL, as illustrated in the following figure. Note that H5FD_STREAM is not available with 1.8.x and later versions of the library.
![](https://support.hdfgroup.org/documentation/hdf5/latest/VFL_Drivers.gif) I/O path from application to VFL and low-level drivers to storage  
---  
As mentioned above, HDF5 applications access HDF5 files through various low-level file drivers. The default driver for that layout is the POSIX driver (also known as the SEC2 driver), [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606). Alternative layouts and drivers are designed to suit the needs of a variety of systems, environments, and applications. The drivers are listed in the table below.
### 
Supported file drivers
Supported file drivers Driver Name  | Driver Identifier  | Description  | Related API   
---|---|---|---  
POSIX  |  [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) | This driver uses POSIX file-system functions like read and write to perform I/O to a single, permanent file on local disk with no system buffering. This driver is POSIX-compliant and is the default file driver for all systems.  |  [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.")  
Memory  |  [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) | With this driver, an application can work with a file in memory for faster reads and writes. File contents are kept in memory until the file is closed. At closing, the memory version of the file can be written back to disk or abandoned.  |  [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.")  
Log  |  [H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4) | This is the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver with logging capabilities.  |  [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.")  
Family  |  [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6) | With this driver, the HDF5 file's address space is partitioned into pieces and sent to separate storage files using an underlying driver of the user's choice. This driver is for systems that do not support files larger than 2 gigabytes.  |  [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.")  
Multi  |  [H5FD_MULTI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmulti_8h.html#a754e05ae5e0f2d86f64002b338c0fd5c) | With this driver, data can be stored in multiple files according to the type of the data. I/O might work better if data is stored in separate files based on the type of data. The Split driver is a special case of this driver.  |  [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") / [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.")  
STDIO  |  [H5FD_STDIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dstdio_8h.html#a030a03b96a9f6e46035ce64e25389085) | This driver uses functions from the standard C stdio.h to perform I/O to a single, permanent file on local disk with additional system buffering.  |  [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.")  
Split  |  [H5FD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#ac6c45c6a8e1cb7f5b4400d95bf651eae) | This file driver splits a file into two parts. One part stores metadata, and the other part stores raw data. This splitting a file into two parts is a limited case of the Multi driver.  |  [H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.")  
Parallel  |  [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) | This is the standard HDF5 file driver for parallel file systems. This driver uses the MPI standard for both communication and file I/O.  |  [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")  
Direct  |  [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436) | This is the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver except data is written to or read from the file synchronously without being cached by the system.  |  [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.")  
Mirror  |  [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) | Serial I/O to file using Unix “stdio” functions.  |  [H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.")  
HDFS  |  [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) | Read-Only access to Hadoop Distributed File System (HDFS).  |  [H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.")  
ros3  |  [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) | Read-Only access to Amazon's S3 service.  |  [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.")  
Subfiling  |  [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) | Derived from other "stacked" VFDs such as the splitter, mirror, and family VFDs.  |  [H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.")  
IOC  |  [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) | Relays VFD calls to one VFD, and write calls to another VFD. Maintains two files.  |  [H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.")  
Onion  |  [H5FD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a1d6673897b4ebd1bad9846b5695ba346) | Provide in-file provenance and revision/version control.  |  [H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list")  
Windows  |  [H5FD_WINDOWS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dwindows_8h.html#ab5173993ddefd103bfb3d37c2837a9a4) | This driver was modified in HDF5-1.8.8 to be a wrapper of the POSIX driver, [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606). This change should not affect user applications.  |  [H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.")  
Parallel POSIX  | H5FD_MPIPOSIX  | This driver is no longer available  |   
Stream  | H5FD_STREAM  | This driver is no longer available.  |   
For more information, see the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) entries for the function calls shown in the column on the right in the table above.
Note that the low-level file drivers manage alternative file storage layouts. Dataset storage layouts (chunking, compression, and external dataset storage) are managed independently of file storage layouts.
If an application requires a special-purpose low-level driver, the VFL provides a public API for creating one. For more information on how to create a driver, see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html) in the HDF5 [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html).
###  Identifying the Previously‐used File Driver
When creating a new HDF5 file, no history exists, so the file driver must be specified if it is to be other than the default.
When opening existing files, however, the application may need to determine which low-level driver was used to create the file. The function [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") is used for this purpose. See the example below.
_Identifying a driver_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8)
hid_t H5Pget_driver(hid_t plist_id)
Returns low-lever driver identifier.
[H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") returns a constant identifying the low-level driver for the access property list fapl_id. For example, if the file was created with the POSIX (aka SEC2) driver, [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") returns [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606).
If the application opens an HDF5 file without both determining the driver used to create the file and setting up the use of that driver, the HDF5 Library will examine the superblock and the driver definition block to identify the driver. See the [HDF5 File Format Specification Version 4.0](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html) for detailed descriptions of the superblock and the driver definition block.
###  The POSIX (aka SEC2) Driver
The POSIX driver, [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606), uses functions from section 2 of the POSIX manual to access unbuffered files stored on a local file system. This driver is also known as the SEC2 driver. The HDF5 Library buffers metadata regardless of the low-level driver, but using this driver prevents data from being buffered again by the lowest layers of the library.
The function [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.") sets the file access properties to use the POSIX driver. See the example below.
_Using the POSIX, aka SEC2, driver_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db)
herr_t H5Pset_fapl_sec2(hid_t fapl_id)
Modifies the file access property list to use the H5FD_SEC2 driver.
Any previously-defined driver properties are erased from the property list.
Additional parameters may be added to this function in the future. Since there are no additional variable settings associated with the POSIX driver, there is no H5Pget_fapl_sec2 function.
###  The Direct Driver
The Direct driver, [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436), functions like the POSIX driver except that data is written to or read from the file synchronously without being cached by the system.
The functions [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.") and [H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d "Retrieves direct I/O driver settings.") are used to manage file access properties. See the example below.
_Using the Direct driver_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t alignment, size_t block_size, size_t cbuf_size)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *alignment, size_t *block_size, size_t *cbuf_size)
[H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c)
herr_t H5Pset_fapl_direct(hid_t fapl_id, size_t alignment, size_t block_size, size_t cbuf_size)
Sets up use of the direct I/O driver.
[H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d)
herr_t H5Pget_fapl_direct(hid_t fapl_id, size_t *boundary, size_t *block_size, size_t *cbuf_size)
Retrieves direct I/O driver settings.
[H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.") sets the file access properties to use the Direct driver; any previously defined driver properties are erased from the property list. [H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d "Retrieves direct I/O driver settings.") retrieves the file access properties used with the Direct driver. fapl_id is the file access property list identifier. alignment is the memory alignment boundary. block_size is the file system block size. cbuf_size is the copy buffer size.
Additional parameters may be added to this function in the future.
###  The Log Driver
The Log driver, [H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4), is designed for situations where it is necessary to log file access activity.
The function [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.") is used to manage logging properties. See the example below.
_Logging file access_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const char *logfile, unsigned int flags, size_t buf_size)
[H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d)
herr_t H5Pset_fapl_log(hid_t fapl_id, const char *logfile, unsigned long long flags, size_t buf_size)
Sets up the logging virtual file driver (H5FD_LOG) for use.
[H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.") sets the file access property list to use the Log driver. File access characteristics are identical to access via the POSIX driver. Any previously defined driver properties are erased from the property list.
Log records are written to the file logfile.
The logging levels set with the verbosity parameter are shown in the table below.
Logging levels Level  | Comments   
---|---  
0  | Performs no logging.   
1  | Records where writes and reads occur in the file.   
2  | Records where writes and reads occur in the file and what kind of data is written at each location. This includes raw data or any of several types of metadata (object headers, superblock, B-tree data, local headers, or global headers).   
There is no H5Pget_fapl_log function.
Additional parameters may be added to this function in the future.
###  The Windows Driver
The Windows driver, [H5FD_WINDOWS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dwindows_8h.html#ab5173993ddefd103bfb3d37c2837a9a4), was modified in HDF5-1.8.8 to be a wrapper of the POSIX driver, [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606). In other words, if the Windows drivers is used, any file I/O will instead use the functionality of the POSIX driver. This change should be transparent to all user applications. The Windows driver used to be the default driver for Windows systems. The POSIX driver is now the default.
The function [H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.") sets the file access properties to use the Windows driver. See the example below.
_Using the Windows driver_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e)
herr_t H5Pset_fapl_windows(hid_t fapl_id)
Sets the Windows I/O driver.
Any previously-defined driver properties are erased from the property list.
Additional parameters may be added to this function in the future. Since there are no additional variable settings associated with the POSIX driver, there is no H5Pget_fapl_windows function.
###  The STDIO Driver
The STDIO driver, [H5FD_STDIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dstdio_8h.html#a030a03b96a9f6e46035ce64e25389085), accesses permanent files in a local file system like the POSIX driver does. The STDIO driver also has an additional layer of buffering beneath the HDF5 Library.
The function [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.") sets the file access properties to use the STDIO driver. See the example below.
_Using the STDIO driver_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
Any previously defined driver properties are erased from the property list.
Additional parameters may be added to this function in the future. Since there are no additional variable settings associated with the STDIO driver, there is no H5Pget_fapl_stdio function.
###  The Memory (aka Core) Driver
There are several situations in which it is reasonable, sometimes even required, to maintain a file entirely in system memory. You might want to do so if, for example, either of the following conditions apply: 
  * Performance requirements are so stringent that disk latency is a limiting factor 
  * You are working with small, temporary files that will not be retained and, thus, need not be written to storage media


The Memory driver, [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a), provides a mechanism for creating and managing such in memory files. The functions [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.") and [H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50 "Queries core file driver properties.") manage file access properties. See the example below.
_Managing file access for in-memory files_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_properties, size_t block_size, bool backing_store)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_properties, size_t *block_size, bool *backing_store)
[H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f)
herr_t H5Pset_fapl_core(hid_t fapl_id, size_t increment, bool backing_store)
Modifies the file access property list to use the H5FD_CORE driver.
[H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50)
herr_t H5Pget_fapl_core(hid_t fapl_id, size_t *increment, bool *backing_store)
Queries core file driver properties.
[H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.") sets the file access property list to use the Memory driver; any previously defined driver properties are erased from the property list.
Memory for the file will always be allocated in units of the specified block_size.
The backing_store Boolean flag is set when the in-memory file is created. backing_store indicates whether to write the file contents to disk when the file is closed. If backing_store is set to 1 (true), the file contents are flushed to a file with the same name as the in-memory file when the file is closed or access to the file is terminated in memory. If backing_store is set to 0 (false), the file is not saved.
The application is allowed to open an existing file with the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver. While using [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") to open an existing file, if backing_store is set to 1 and the flag for [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), changes to the file contents will be saved to the file when the file is closed. If backing_store is set to 0 and the flag for [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), changes to the file contents will be lost when the file is closed. If the flag for [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), no change to the file will be allowed either in memory or on file.
If the file access property list is set to use the Memory driver, [H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50 "Queries core file driver properties.") will return block_size and backing_store with the relevant file access property settings.
Note the following important points regarding in-memory files: 
  * Local temporary files are created and accessed directly from memory without ever being written to disk 
  * Total file size must not exceed the available virtual memory 
  * Only one HDF5 file identifier can be opened for the file, the identifier returned by [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.")
  * The changes to the file will be discarded when access is terminated unless backing_store is set to 1


Additional parameters may be added to these functions in the future.
see [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html) section for information on more advanced usage of the Memory file driver, and see [Modified Region Writes](https://support.hdfgroup.org/releases/hdf5/documentation/advanced_topics/ModifiedRegionWrites.pdf) section for information on how to set write operations so that only modified regions are written to storage.
###  The Family Driver
HDF5 files can become quite large, and this can create problems on systems that do not support files larger than 2 gigabytes. The HDF5 file family mechanism is designed to solve the problems this creates by splitting the HDF5 file address space across several smaller files. This structure does not affect how metadata and raw data are stored: they are mixed in the address space just as they would be in a single, contiguous file.
HDF5 applications access a family of files via the Family driver, [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6). The functions [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.") and [H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375 "Returns file access property list information.") are used to manage file family properties. See the example below.
_Managing file family properties_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id,
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) member_properties)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id,
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *member_properties)
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)
herr_t H5Pset_fapl_family(hid_t fapl_id, hsize_t memb_size, hid_t memb_fapl_id)
Sets the file access property list to use the family driver.
[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375)
herr_t H5Pget_fapl_family(hid_t fapl_id, hsize_t *memb_size, hid_t *memb_fapl_id)
Returns file access property list information.
Each member of the family is the same logical size though the size and disk storage reported by file system listing tools may be substantially smaller. Examples of file system listing tools are 
ls -l
on a Unix system or the detailed folder listing on an Apple or Microsoft Windows system. The name passed to [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") should include a printf(3c)-style integer format specifier which will be replaced with the family member number. The first family member is numbered zero (0).
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.") sets the access properties to use the Family driver; any previously defined driver properties are erased from the property list. member_properties will serve as the file access property list for each member of the file family. memb_size specifies the logical size, in bytes, of each family member. memb_size is used only when creating a new file or truncating an existing file; otherwise the member size is determined by the size of the first member of the family being opened.
[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375 "Returns file access property list information.") is used to retrieve file family properties. If the file access property list is set to use the Family driver, member_properties will be returned with a pointer to a copy of the appropriate member access property list. If memb_size is non-null, it will contain the logical size, in bytes, of family members.
Additional parameters may be added to these functions in the future.
#### Unix Tools and an HDF5 Utility
It occasionally becomes necessary to repartition a file family. A command-line utility for this purpose, [h5repart](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_t__u_g.html#sec_cltools_h5repart), is distributed with the HDF5 library.
h5repart [-v] [-b block_size[suffix]] [-m member_size[suffix]] source destination
[h5repart](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_t__u_g.html#sec_cltools_h5repart) repartitions an HDF5 file by copying the source file or file family to the destination file or file family, preserving holes in the underlying UNIX files. Families are used for the source and/or destination if the name includes a printf-style integer format such as d. The -v switch prints input and output file names on the standard error stream for progress monitoring, -b sets the I/O block size (the default is 1KB), and -m sets the output member size if the destination is a family name (the default is 1GB). block_size and member_size may be suffixed with the letters g, m, or k for GB, MB, or KB respectively.
The [h5repart](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_t__u_g.html#sec_cltools_h5repart) utility is described on the Tools page of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
An existing HDF5 file can be split into a family of files by running the file through split(1) on a UNIX system and numbering the output files. However, the HDF5 Library is lazy about extending the size of family members, so a valid file cannot generally be created by concatenation of the family members.
Splitting the file and rejoining the segments by concatenation (split(1) and cat(1) on UNIX systems) does not generate files with holes; holes are preserved only through the use of [h5repart](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_t__u_g.html#sec_cltools_h5repart).
###  The Multi Driver
In some circumstances, it is useful to separate metadata from raw data and some types of metadata from other types of metadata. Situations that would benefit from use of the Multi driver include the following: 
  * In networked situations where the small metadata files can be kept on local disks but larger raw data files must be stored on remote media 
  * In cases where the raw data is extremely large 
  * In situations requiring frequent access to metadata held in RAM while the raw data can be efficiently held on disk


In either case, access to the metadata is substantially easier with the smaller, and possibly more localized, metadata files. This often results in improved application performance.
The Multi driver, [H5FD_MULTI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmulti_8h.html#a754e05ae5e0f2d86f64002b338c0fd5c), provides a mechanism for segregating raw data and different types of metadata into multiple files. The functions [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") and [H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d "Returns information about the multi-file access property list.") are used to manage access properties for these multiple files. See the example below.
_Managing access properties for multiple files_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) *memb_map, const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl,
const char * const *memb_name, const [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *memb_addr,
bool relax)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) *memb_map, const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl,
const char **memb_name, const [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *memb_addr, bool *relax)
[H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b)
enum H5F_mem_t H5FD_mem_t
**Definition** H5FDpublic.h:273
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396)
herr_t H5Pset_fapl_multi(hid_t fapl_id, const H5FD_mem_t *memb_map, const hid_t *memb_fapl, const char *const *memb_name, const haddr_t *memb_addr, bool relax)
Sets up use of the multi-file driver.
[H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d)
herr_t H5Pget_fapl_multi(hid_t fapl_id, H5FD_mem_t *memb_map, hid_t *memb_fapl, char **memb_name, haddr_t *memb_addr, bool *relax)
Returns information about the multi-file access property list.
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") sets the file access properties to use the Multi driver; any previously defined driver properties are erased from the property list. With the Multi driver invoked, the application will provide a base name to [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") or [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file."). The files will be named by that base name as modified by the rule indicated in memb_name. File access will be governed by the file access property list memb_properties.
See [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") and [H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d "Returns information about the multi-file access property list.") in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) for descriptions of these functions and their usage.
Additional parameters may be added to these functions in the future.
###  The Split Driver
The Split driver is a limited case of the Multi driver where only two files are created. One file holds metadata, and the other file holds raw data. The function [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.") is used to manage Split file access properties. See the example below.
_Managing access properties for split files_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_properties, const char *meta_extension,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) meta_properties,const char *raw_extension, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) raw_properties)
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)
herr_t H5Pset_fapl_split(hid_t fapl, const char *meta_ext, hid_t meta_plist_id, const char *raw_ext, hid_t raw_plist_id)
Emulates the old split file driver.
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.") sets the file access properties to use the Split driver; any previously defined driver properties are erased from the property list.
With the Split driver invoked, the application will provide a base file name such as file_name to [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."). The metadata and raw data files in storage will then be named file_name.meta_extension and file_name.raw_extension, respectively. For example, if meta_extension is defined as .meta and raw_extension is defined as .raw, the final filenames will be file_name.meta and file_name.raw.
Each file can have its own file access property list. This allows the creative use of other lowlevel file drivers. For instance, the metadata file can be held in RAM and accessed via the Memory driver while the raw data file is stored on disk and accessed via the POSIX driver. Metadata file access will be governed by the file access property list in meta_properties. Raw data file access will be governed by the file access property list in raw_properties.
Additional parameters may be added to these functions in the future. Since there are no additional variable settings associated with the Split driver, there is no H5Pget_fapl_split function.
###  The ROS3 Driver
The ROS3 (read-only S3) driver is used to enable read-only access to HDF5 files which are stored in Amazon's S3 web service (<https://aws.amazon.com/s3/>) or an S3 API-compatible storage system. This driver expects that HDF5 files will be stored in the storage system as a single object. It translates I/O read requests from the HDF5 library into the appropriate REST API requests that will retrieve the relevant parts of the HDF5 file needed by the read requests.
The functions [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.") and [H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties.") are used to manage file access properties for the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver. See the example below.
_Managing access properties for ROS3_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_properties, const [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) *fa)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) *fa_out)
typedef struct [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) {
int32_t [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a67fae7dd1de9edce3656ed214d20377f);
bool [authenticate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ad3dab4b20a1b92a3ddae467c54f381d3);
char [aws_region](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a7378f9932dad7faf06da15d5d47a664e)[[H5FD_ROS3_MAX_REGION_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#aeaebf5218d3c900eb9a8d981f18a5bd6) + 1];
char [secret_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a30a5353319d0a23ebc0b347aff48544a)[[H5FD_ROS3_MAX_SECRET_ID_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a0dfcbaac69a09596185e619bc9f1c460) + 1];
char [secret_key](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ab622227877522d00a1284b9e5465d02c)[[H5FD_ROS3_MAX_SECRET_KEY_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a6492f232f64a036290090e7cdf2f9161) + 1];
} [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html);
[H5FD_ROS3_MAX_SECRET_ID_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a0dfcbaac69a09596185e619bc9f1c460)
#define H5FD_ROS3_MAX_SECRET_ID_LEN
**Definition** H5FDros3.h:58
[H5FD_ROS3_MAX_SECRET_KEY_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a6492f232f64a036290090e7cdf2f9161)
#define H5FD_ROS3_MAX_SECRET_KEY_LEN
**Definition** H5FDros3.h:64
[H5FD_ROS3_MAX_REGION_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#aeaebf5218d3c900eb9a8d981f18a5bd6)
#define H5FD_ROS3_MAX_REGION_LEN
**Definition** H5FDros3.h:52
[H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f)
herr_t H5Pset_fapl_ros3(hid_t fapl_id, const H5FD_ros3_fapl_t *fa)
Modifies the specified File Access Property List to use the H5FD_ROS3 driver.
[H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e)
herr_t H5Pget_fapl_ros3(hid_t fapl_id, H5FD_ros3_fapl_t *fa_out)
Queries a File Access Property List for H5FD_ROS3 file driver properties.
[H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html)
Configuration structure for H5Pset_fapl_ros3() / H5Pget_fapl_ros3().
**Definition** H5FDros3.h:216
[H5FD_ros3_fapl_t::secret_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a30a5353319d0a23ebc0b347aff48544a)
char secret_id[128+1]
**Definition** H5FDros3.h:220
[H5FD_ros3_fapl_t::version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a67fae7dd1de9edce3656ed214d20377f)
int32_t version
**Definition** H5FDros3.h:217
[H5FD_ros3_fapl_t::aws_region](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a7378f9932dad7faf06da15d5d47a664e)
char aws_region[32+1]
**Definition** H5FDros3.h:219
[H5FD_ros3_fapl_t::secret_key](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ab622227877522d00a1284b9e5465d02c)
char secret_key[128+1]
**Definition** H5FDros3.h:221
[H5FD_ros3_fapl_t::authenticate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ad3dab4b20a1b92a3ddae467c54f381d3)
bool authenticate
**Definition** H5FDros3.h:218
[H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.") sets the file access properties managed by the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver and [H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties.") retrieves those file access properties.
The `version` field is used to specify the revision of the [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html "Configuration structure for H5Pset_fapl_ros3\(\) / H5Pget_fapl_ros3\(\).") structure and currently must always be set to the macro value [H5FD_CURR_ROS3_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a0370e0bade82fd8ff6ac012644599fe2).
The `authenticate` field is a boolean determining whether or not the driver should use credentials specified in the `secret_id` and `secret_key` fields of the structure. If `true`, credentials will be used from the structure. If `false`, the driver will instead look for credentials from standard AWS environment variables, configuration files and other locations. In this case, any values specified in the `secret_id` and `secret_key` fields will be ignored.
The `aws_region` field is used to specify the AWS region to use when accessing files opened through the File Access Property List with these properties set on it. If this field is an empty string, the driver will search for a specified AWS region in the standard AWS environment variables and configuration files. An AWS region _must_ be specified with one of these mechanisms or an error will be returned when attempting to open a file with the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver.
###  The Parallel Driver
Parallel environments require a parallel low-level driver. HDF5's default driver for parallel systems is called the Parallel driver, [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf). This driver uses the MPI standard for both communication and file I/O.
The functions [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") and [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") are used to manage file access properties for the [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) driver. See the example below.
_Managing parallel file access properties_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm comm, MPI_info info)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm *comm, MPI_info *info)
[H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a)
herr_t H5Pget_fapl_mpio(hid_t fapl_id, MPI_Comm *comm, MPI_Info *info)
Returns MPI IO communicator information.
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)
herr_t H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)
Stores MPI IO communicator information to the file access property list.
The file access properties managed by [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") and retrieved by [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") are the MPI communicator, comm, and the MPI info object, info. comm and info are used for file open. info is an information object much like an HDF5 property list. Both are defined in MPI_FILE_OPEN of MPI.
The communicator and the info object are saved in the file access property list fapl_id. fapl_id can then be passed to MPI_FILE_OPEN to create and/or open the file.
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") and [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") are available only in the parallel HDF5 Library and are not collective functions. The Parallel driver is available only in the parallel HDF5 Library.
Additional parameters may be added to these functions in the future.
##  Code Examples for Opening and Closing Files
###  Example Using the H5F_ACC_TRUNC Flag
The following example uses the [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) flag when it creates a new file. The default file creation and file access properties are also used. Using [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) means the function will look for an existing file with the name specified by the function. In this case, that name is FILE. If the function does not find an existing file, it will create one. If it does find an existing file, it will empty the file in preparation for a new set of data. The identifier for the "new" file will be passed back to the application program. For more information,  

See also
     [File Access Modes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_access_modes).
_Creating a file with default creation and access properties_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file; // identifier
// Create a new file using H5F_ACC_TRUNC access, default
// file creation properties, and default file access
// properties.
file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(FILE, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Close the file.
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
###  Example with the File Creation Property List
The example below shows how to create a file with 64-bit object offsets and lengths.
_Creating a file with 64-bit offsets_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) create_plist;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id;
create_plist = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977));
[H5Pset_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae5eb3ba16f063d151d1b56d33e0710a9)(create_plist, 8, 8);
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(“test.h5”, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), create_plist, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
.
.
.
H5Fclose(file_id);
[H5Pset_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae5eb3ba16f063d151d1b56d33e0710a9)
herr_t H5Pset_sizes(hid_t plist_id, size_t sizeof_addr, size_t sizeof_size)
Sets the byte size of the offsets and lengths used to address objects in an HDF5 file.
###  Example with the File Access Property List
This example shows how to open an existing file for independent datasets access by MPI parallel I/O:
_Opening an existing file for parallel I/O_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_plist;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id;
access_plist = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
H5Pset_fapl_mpi(access_plist, MPI_COMM_WORLD, MPI_INFO_NULL);
// H5Fopen must be called collectively
file_id = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(“test.h5”, [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), access_plist);
.
.
.
// H5Fclose must be called collectively
H5Fclose(file_id);
[H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)
#define H5F_ACC_RDWR
**Definition** H5Fpublic.h:29
##  Working with Multiple HDF5 Files
Multiple HDF5 files can be associated so that the files can be worked with as though all the information is in a single HDF5 file. A temporary association can be set up by means of the [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") function. A permanent association can be set up by means of the external link function [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.").
The purpose of this section is to describe what happens when the [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") function is used to mount one file on another.
When a file is mounted on another, the mounted file is mounted at a group, and the root group of the mounted file takes the place of that group until the mounted file is unmounted or until the files are closed.
The figure below shows two files before one is mounted on the other. File1 has two groups and three datasets. The group that is the target of the A link has links, Z and Y, to two of the datasets. The group that is the target of the B link has a link, W, to the other dataset. File2 has three groups and three datasets. The groups in File2 are the targets of the AA, BB, and CC links. The datasets in File2 are the targets of the ZZ, YY, and WW links.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Files_fig3.gif) Two separate files  
---  
The figure below shows the two files after File2 has been mounted File1 at the group that is the target of the B link.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Files_fig4.gif) File2 mounted on File1  
---  
Note: In the figure above, the dataset that is the target of the W link is not shown. That dataset is masked by the mounted file.
If a file is mounted on a group that has members, those members are hidden until the mounted file is unmounted. There are two ways around this if you need to work with a group member. One is to mount the file on an empty group. Another is to open the group member before you mount the file. Opening the group member will return an identifier that you can use to locate the group member.
The example below shows how [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") might be used to mount File2 onto File1.
_Using H5Fmount_
status = [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083)(loc_id, "/B", child_id, plist_id)
[H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083)
herr_t H5Fmount(hid_t loc_id, const char *name, hid_t child, hid_t plist)
Mounts an HDF5 file.
Note: In the code example above, loc_id is the file identifier for File1, /B is the link path to the group where File2 is mounted, child_id is the file identifier for File2, and plist_id is a property list identifier. For more information,  

See also
     [HDF5 Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#sec_group).
See the entries for [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file."), [H5Funmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga79e80cbf6d9782402b353b763162779c "Un-mounts an HDF5 file."), and [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
Previous Chapter [The HDF5 Library and Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#sec_program) - Next Chapter [HDF5 Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#sec_group)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


