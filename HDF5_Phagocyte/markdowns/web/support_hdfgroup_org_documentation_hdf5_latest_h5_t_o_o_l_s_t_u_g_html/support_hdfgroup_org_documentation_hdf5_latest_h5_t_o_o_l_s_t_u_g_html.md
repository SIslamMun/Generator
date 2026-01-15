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
  * [ h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#title2 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#title3 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5stat Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5stat
##  Introduction
With h5stat, you can dump stats from an HDF5 file.
##  Usage
#### h5stat [OPTIONS] file
##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–version** Print the library version number and exit 
  * **–file** Print file information 
  * **–filemetadata** Print file space information for file's metadata 
  * **–group** Print group information 
  * **–links=N** Set the threshold for the # of links when printing information for small groups. N is an integer greater than 0. The default threshold is 10. 
  * **–groupmetadata** Print file space information for groups' metadata 
  * **–dset** Print dataset information 
  * **–dims=N** Set the threshold for the dimension sizes when printing information for small datasets. N is an integer greater than 0. The default threshold is 10. 
  * **–dsetmetadata** Print file space information for datasets' metadata 
  * **–dtypemetadata** Print datasets' datatype information 
  * **–attribute** Print attribute information 
  * **–numattrs=N** Set the threshold for the number of attributes when printing information for small number of attributes. N is an integer greater than 0. The default threshold is 10. 
  * **–freespace** Print free space information 
  * **–summary** Print summary of file space information 
  * **–page-buffer-size=N** Set the page buffer cache size, N=non-negative integers 
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
(\<namenode name\>,\<namenode port\>,
...\<kerberos cache path\>,\<username\>,\<buffer size\>) 
If absent or A == 
(,,,,) 
all default values are used. Has no effect if vfd flag is not 'hdfs'.  
If an attribute is empty, a default value will be used. 
  * **–vol-value** Value (ID) of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-name** Name of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-info** VOL-specific info to pass to the VOL connector used for opening the HDF5 file specified If none of the above options are used to specify a VOL, then the VOL named by HDF5_VOL_CONNECTOR (or the native VOL connector, if that environment variable is unset) will be used 
  * **–vfd-value** Value (ID) of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-name** Name of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-info** VFD-specific info to pass to the VFL driver used for opening the HDF5 file specified


Previous Chapter [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) - Next Chapter [h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#sec_cltools_h5clear)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


