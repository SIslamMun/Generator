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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title0 "H1")
  * [ Error Messages](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title1 "H2")
  * [ Invariant Conditions](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title2 "H2")
  * [ Timings and Statistics](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title3 "H2")
  * [ API Tracing](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title4 "H2")
  * [ Error Messages](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title5 "H1")
  * [ Invariant Conditions](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title6 "H1")
  * [ Timings and Statistics](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title7 "H1")
  * [ Sample debug specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_a_p_p_d_b_g.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Debugging HDF5 Applications
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction
The HDF5 library contains a number of debugging features to make programmers' lives easier including the ability to print detailed error messages, check invariant conditions, display timings and other statistics.
##  Error Messages
Error messages are normally displayed automatically on the standard error stream and include a stack trace of the library including file names, line numbers, and function names. The application has complete control over how error messages are displayed and can disable the display on a permanent or temporary basis. Refer to the documentation for the H5E error handling package.
##  Invariant Conditions
Unless NDEBUG is defined during compiling, the library will include code to verify that invariant conditions have the expected values. When a problem is detected the library will display the file and line number within the library and the invariant condition that failed. A core dump may be generated for post mortem debugging. The code to perform these checks can be included on a per-package bases.
##  Timings and Statistics
The library can be configured to accumulate certain statistics about things like cache performance, datatype conversion, data space conversion, and data filters. The code is included on a per-package basis and enabled at runtime by an environment variable.
##  API Tracing
Tracing API calls has been removed from the library.
The statistics can be displayed on any output stream (including streams opened by the shell) with output from different packages even going to different streams.
#  Error Messages
By default any API function that fails will print an error stack to the standard error stream. 
HDF5-DIAG: Error detected in thread 0. Back trace follows.
#000: H5F.c line 1245 in [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(): unable to open file
major(04): File interface
minor(10): Unable to open file
#001: H5F.c line 846 in H5F_open(): file does not exist
major(04): File interface
minor(10): Unable to open file
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
The error handling package (H5E) is described elsewhere.
#  Invariant Conditions
To include checks for invariant conditions the library should be configured with –disable-production, the default for versions before 1.2. The library designers have made every attempt to handle error conditions gracefully but an invariant condition assertion may fail in certain cases. The output from a failure usually looks something like this: 
Assertion failed: [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html).c:123: i<NELMTS(H5_debug_g)
IOT Trap, core dumped.
[H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)
**Definition** H5AbstractDs.cpp:33
#  Timings and Statistics
Code to accumulate statistics is included at compile time by using the –enable-debug configure switch. The switch can be followed by an equal sign and a comma-separated list of package names or else a default list is used. 
**Name** |  **Default** |  **Description**  
---|---|---  
a | No | Attributes   
ac | Yes | Meta data cache   
b | Yes | B-Trees   
d | Yes | Datasets   
e | Yes | Error handling   
f | Yes | Files   
g | Yes | Groups   
hg | Yes | Global heap   
hl | No | Local heaps   
i | Yes | Interface abstraction   
mf | No | File memory management   
mm | Yes | Library memory management   
o | No | Object headers and messages   
p | Yes | Property lists   
s | Yes | Data spaces   
t | Yes | Datatypes   
v | Yes | Vectors   
z | Yes | Raw data filters   
In addition to including the code at compile time the application must enable each package at runtime. This is done by listing the package names in the HDF5_DEBUG environment variable. That variable may also contain file descriptor numbers (the default is '2') which control the output for all following packages up to the next file number. The word 'all' refers to all packages. Any word my be preceded by a minus sign to turn debugging off for the package.
##  Sample debug specifications
all  | This causes debugging output from all packages to be sent to the standard error stream.   
---|---  
all -t -s  | Debugging output for all packages except datatypes and data spaces will appear on the standard error stream.   
-all ac 255 t,s  | This disables all debugging even if the default was to debug something, then output from the meta data cache is send to the standard error stream and output from data types and spaces is sent to file descriptor 255 which should be redirected by the shell.   
The components of the HDF5_DEBUG value may be separated by any non-lowercase letter.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


