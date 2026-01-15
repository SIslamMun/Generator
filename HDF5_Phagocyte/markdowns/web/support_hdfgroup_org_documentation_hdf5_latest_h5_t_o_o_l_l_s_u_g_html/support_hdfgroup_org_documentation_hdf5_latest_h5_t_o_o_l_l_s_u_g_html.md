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
  * [ h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title2 "H2")
  * [ file/OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title3 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title4 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5ls Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5ls
##  Introduction
With h5ls, you can dump objects from an HDF5 file.
##  Usage
#### h5ls [OPTIONS] file[/OBJECT] [file[/[OBJECT]...]
##  file/OBJECT
Each object consists of an HDF5 file name optionally followed by a slash and an object name within the file (if no object is specified within the file then the contents of the root group are displayed). The file name may include a printf(3C) integer format such as "%05d" to open a file family.
##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–address** Print raw data address. If dataset is contiguous, address is offset in file of beginning of raw data. If chunked, returned list of addresses indicates offset of each chunk. Must be used with **–verbose option**. Provides no information for non-dataset objects. 
  * **–data** Print the values of datasets 
  * **–follow-symlinks** Follow symbolic links (soft links and external links) to display target object information.  
Without this option, h5ls identifies a symbolic link as a soft link or external link and prints the value assigned to the symbolic link; it does not provide any information regarding the target object or determine whether the link is a dangling link. 
  * **–no-dangling-links** Must be used with **–follow-symlinks** option; otherwise, h5ls shows error message and returns an exit code of 1.  
Check for any symbolic links (soft links or external links) that do not resolve to an existing object (dataset, group, or named datatype).  
If any dangling link is found, this situation is treated as an error and h5ls returns an exit code of 1. 
  * **–full** Print full path names instead of base names 
  * **–group** Show information about a group, not its contents 
  * **–label** Label members of compound datasets 
  * **–recursive** List all groups recursively, avoiding cycles 
  * **–string** Print 1-byte integer datasets as ASCII 
  * **–simple** Use a machine-readable output format 
  * **–width=N** Set the number of columns of output 
  * **–verbose** Generate more verbose output 
  * **–version** Print the library version number and exit 
  * **–page-buffer-size=N** Set the page buffer cache size, N=non-negative integers 
  * **–vfd=DRIVER** Use the specified virtual file driver 
  * **–hexdump** Show raw data in hexadecimal format 
  * **–endpoint-url=P** Supply S3 endpoint url information to "ros3" vfd. P is the AWS service endpoint. Has no effect if vfd flag not set to "ros3". 
  * **–s3-cred=C** Supply S3 authentication information to "ros3" vfd. Accepts tuple of 
(<aws-region>,<access-id>,<access-key>) 
or 
(<aws-region>,<access-id>,<access-key>,<session-token>) 
. If absent or C = 
(,,) 
or C = 
(,,,) 
defaults to no-authentication. Has no effect if vfd flag not set to "ros3". 
  * **–hdfs-attrs=A** Supply configuration information to Hadoop VFD. Accepts tuple of 
(<namenode name>,<namenode port>,
...<kerberos cache path>,<username>,<buffer size>) 
If absent or A == 
(,,,,) 
all default values are used. Has no effect if vfd flag is not 'hdfs'. 
  * **–vol-value** Value (ID) of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-name** Name of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-info** VOL-specific info to pass to the VOL connector used for opening the HDF5 file specified If none of the above options are used to specify a VOL, then the VOL named by HDF5_VOL_CONNECTOR (or the native VOL connector, if that environment variable is unset) will be used 
  * **–vfd-value** Value (ID) of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-name** Name of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-info** VFD-specific info to pass to the VFL driver used for opening the HDF5 file specified


###  Deprecated Options
The following options have been removed in HDF5 1.12. Use the indicated replacement option in all work. 
  * **–external** Follow external links.  
Replaced by **–follow-symlinks**. 
  * **–errors** Show all HDF5 error reporting  
Replaced by **–enable-error-stack**.


Previous Chapter [h5jam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#sec_cltools_h5jam) - Next Chapter [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


