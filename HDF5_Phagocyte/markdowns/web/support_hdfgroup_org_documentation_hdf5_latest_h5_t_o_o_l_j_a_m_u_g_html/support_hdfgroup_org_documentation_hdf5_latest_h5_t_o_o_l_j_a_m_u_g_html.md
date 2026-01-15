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
  * [ h5jam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title2 "H2")
  * [ h5jam Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title3 "H2")
  * [ h5unjam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title4 "H1")
  * [ h5unjam Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5jam/h5unjam Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5jam
##  Introduction
  * h5jam Adds user block to the front of an HDF5 file and creates a new concatenated file. 
  * h5unjam Splits an HDF5 file into two files, one containing the user block data and the other the HDF5 data.


##  Usage
#### h5jam -i <in_file.h5> -u <in_user_file> [-o <out_file.h5>] [–clobber]
#### h5unjam -i <in_file.h5> [-o <out_file.h5> ] [-u <out_user_file> | –delete]
##  h5jam Options
  * **-i in_file.h5** Specifies the input HDF5 file. 
  * **-u in_user_file** Specifies the file to be inserted into the user block. Can be any file format except an HDF5 format. 
  * **-o out_file.h5** Specifies the output HDF5 file. If not specified, the user block will be concatenated in place to the input HDF5 file. 
  * **–clobber** Wipes out any existing user block before concatenating the given user block. The size of the new user block will be the larger of:
    * the size of existing user block in the input HDF5 file
    * the size of user block required by new input user file (size = 512 x 2N, N is positive integer.) 
  * **-h** Prints a usage message and exits. 
  * **-V** Prints the HDF5 library version and exits.


#  h5unjam
##  h5unjam Options
  * **-i in_file.h5** Specifies the HDF5 as input. If the input HDF5 file contains no user block, exit with an error message. 
  * **-o out_file.h5** Specifies output HDF5 file without a user block. If not specified, the user block will be removed from the input HDF5 file. 
  * **-u out_user_file** Specifies the output file containing the data from the user block. Cannot be used with –delete option. 
  * **–delete** Remove the user block from the input HDF5 file. The content of the user block is discarded. Cannot be used with the -u option. 
  * **-h** Prints a usage message and exits. 
  * **-V** Prints the HDF5 library version and exits.


If neither **–delete** nor **-u** is specified, the user block from the input file will be displayed to stdout stream.
Previous Chapter [h5import](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#sec_cltools_h5import) - Next Chapter [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


