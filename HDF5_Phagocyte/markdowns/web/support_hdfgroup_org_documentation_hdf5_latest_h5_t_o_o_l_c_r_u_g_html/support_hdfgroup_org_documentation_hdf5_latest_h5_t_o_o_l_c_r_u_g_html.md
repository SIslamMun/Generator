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
  * [ h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title2 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title3 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title4 "H2")
  * [ Usage Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5clear Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5clear
##  Introduction
With h5clear, you can clear the superblock status flag field, remove the metadata cache image, print the EOA and EOF, or set the EOA of a file. It is not a general repair tool and should not be used to fix file corruption. If a process doesn't shut down cleanly, the superblock mark can be left that prevents opening a file without SWMR. Then, h5clear can be used to remove this superblock mark so that the file can be inspected and appropriate actions can be taken.
##  Usage
#### h5clear [OPTIONS] file_name
##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–version** Print the library version number and exit 
  * **–status** Clear the status_flags field in the file's superblock 
  * **–image** Remove the metadata cache image from the file 
  * **–filesize** Print the file's EOA and EOF 
  * **–increment=C** Set the file's EOA to the maximum of (EOA, EOF) + C for the file <file_name>. C is >= 0; C is optional and will default to 1M when not set. This option helps to repair a crashed SWMR file when the stored EOA in the superblock is different from the actual EOF. The file's EOA and EOF will be the same after applying this option to the file.


##  Usage Examples
  * 1) h5clear -s file_name ```
 Clear the status_flags field in the superblock of the HDF5 file <file_name>.

```



  * 2) h5clear -m file_name ```
 Remove the metadata cache image from the HDF5 file <file_name>.

```



  * 3) h5clear –increment file_name ```
 Set the EOA to the maximum of (EOA, EOF) + 1M for the file <file_name>.

```



  * 4) h5clear –increment=512 file_name ```
 Set the EOA to the maximum of (EOA, EOF) + 512 for the file\<file_name>.

```



Previous Chapter [h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#sec_cltools_h5stat) - Next Chapter [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


