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
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title2 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title3 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title4 "H2")
  * [ File Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title5 "H2")
  * [ Object Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title6 "H2")
  * [ Object Property Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title7 "H2")
  * [ Formatting Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title8 "H2")
  * [ XML Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title9 "H2")
  * [ Subsetting Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title10 "H2")
  * [ Usage Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5dump Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5dump
##  Introduction
With h5dump, you can display objects from an HDF5 file.
##  Usage
#### h5dump [OPTIONS] [files
##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–version** Print the library version number and exit


##  File Options
  * **–contents** Print a list of the file contents, group and dataset, names and values, then exit. Optional value 1 also prints attributes, –contents=1. 
  * **–superblock** Print the content of the super block 
  * **–header** Print the header only; no data is displayed 
  * **–filedriver=D** Specify which driver to open the file with 
  * **–output=F** Output raw data into file F 
  * **–binary=B** Binary file output, of form B 
  * **–ddl=F** Output ddl text into file F Use blank(empty) filename F to suppress ddl display 
  * **–page-buffer-size=N** Set the page buffer cache size, N=non-negative integers 
  * **–endpoint-url=P** Supply S3 endpoint url information to "ros3" vfd. P is the AWS service endpoint. Has no effect if filedriver is not "ros3". 
  * **–s3-cred= <cred>** Supply S3 authentication information to "ros3" vfd. 
<cred> :: "(<aws-region>,<access-id>,<access-key>)"
<cred> :: "(<aws-region>,<access-id>,<access-key>,<session-token>)"
If absent or 
\<cred\> -> "(,,)"
or 
\<cred\> -> "(,,,)"
, no authentication. Has no effect if filedriver is not "ros3". 
  * **–hdfs-attrs= <attrs>** Supply configuration information for HDFS file access. For use with **–filedriver=hdfs**
<attrs> :: (<namenode name>,<namenode port>, <kerberos cache
path>,<username>, <buffer size>) 
Any absent attribute will use a default value. 
  * **–vol-value** Value (ID) of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-name** Name of the VOL connector to use for opening the HDF5 file specified 
  * **–vol-info** VOL-specific info to pass to the VOL connector used for opening the HDF5 file specified.  
If none of the above options are used to specify a VOL, then the VOL named by **HDF5_VOL_CONNECTOR** (or the native VOL connector, if that environment variable is unset) will be used 
  * **–vfd-value** Value (ID) of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-name** Name of the VFL driver to use for opening the HDF5 file specified 
  * **–vfd-info** VFD-specific info to pass to the VFL driver used for opening the HDF5 file specified


##  Object Options
  * **–attribute=P** Print the specified attribute If an attribute name contains a slash (/), escape the slash with a preceding backslash (). (See example section below.) 
  * **–dataset=P** Print the specified dataset 
  * **–group=P** Print the specified group and all members 
  * **–soft-link=P** Print the value(s) of the specified soft link 
  * **–datatype=P** Print the specified named datatype 
  * **–any_path=P** Print any attribute, dataset, group, datatype, or link that matches P P can be the absolute path or just a relative path. 
  * **–onlyattr** Print the header and value of attributes Optional value 0 suppresses printing attributes. 
  * **–vds-view-first-missing** Set the VDS bounds to first missing mapped elements. 
  * **–vds-gap-size=N** Set the missing file gap size, N=non-negative integers


##  Object Property Options
  * **–object-ids** Print the object ids 
  * **–properties** Print dataset filters, storage layout and fill value 
  * **–packedbits=L** Print packed bits as unsigned integers, using mask format L for an integer dataset specified with option -d. L is a list of offset,length values, separated by commas. Offset is the beginning bit in the data value and length is the number of bits of the mask. 
  * **–region** Print dataset pointed by region references


##  Formatting Options
  * **–escape** Escape non printing characters 
  * **–string** Print 1-byte integer datasets as ASCII 
  * **–noindex** Do not print array indices with the data 
  * **–format=T** Set the floating point output format 
  * **–lformat=T** Set the floating point long double output format 
  * **–sort_by=Q** Sort groups and attributes by index Q 
  * **–sort_order=Z** Sort groups and attributes by order Z 
  * **–no-compact-subset** Disable compact form of subsetting and allow the use of "[" in dataset names. 
  * **–width=N** Set the number of columns of output. A value of 0 (zero) sets the number of columns to the maximum (65535). Default width is 80 columns.


##  XML Options
  * **–xml** Output in XML using Schema 
  * **–use-dtd** Output in XML using DTD 
  * **–xml-dtd=U** Use the DTD or schema at U 
  * **–xml-ns=S** (XML Schema) Use qualified names n the XML ":": no namespace, default: "hdf5:" E.g., to dump a file called "-f", use h5dump – -f


##  Subsetting Options
Subsetting is available by using the following options with a dataset option. Subsetting is done by selecting a hyperslab from the data. Thus, the options mirror those for performing a hyperslab selection.  
One of the **START** , **COUNT** , **STRIDE** , or **BLOCK** parameters are mandatory if you do subsetting. The **STRIDE** , **COUNT** , and **BLOCK** parameters are optional and will default to 1 in each dimension. **START** is optional and will default to 0 in each dimension.
  * **–start=START** Offset of start of subsetting selection **START** - is a list of integers, the number of which are equal to the number of dimensions in the dataspace being queried.  

  * **–stride=STRIDE** Hyperslab stride **COUNT** - is a list of integers, the number of which are equal to the number of dimensions in the dataspace being queried.  

  * **–count=COUNT** Number of blocks to include in selection **STRIDE** - is a list of integers, the number of which are equal to the number of dimensions in the dataspace being queried.  

  * **–block=BLOCK** Size of block in hyperslab **BLOCK** - is a list of integers, the number of which are equal to the number of dimensions in the dataspace being queried.  
(Alternate compact form of subsetting is described in the Reference Manual)


###  Option Argument Conventions
  * **D** - is the file driver to use in opening the file. Acceptable values are available from <https://support.hdfgroup.org/releases/hdf5/documentation/registered_virtual_file_drivers_vfds.md>. Without the file driver flag, the file will be opened with each driver in turn and in the order specified above until one driver succeeds in opening the file. See examples below for family, split, and multi driver special file name usage.


  * **F** - is a filename. 
  * **P** - is the full path from the root group to the object. 
  * **N** - is an integer greater than 1. 
  * **T** - is a string containing the floating point format, e.g '%.3g' 
  * **T** - is a string containing the floating point long double format, e.g '%.3Lg' 
  * **U** - is a URI reference (as defined in [IETF RFC 2396], updated by [IETF RFC 2732]) 
  * **B** - is the form of binary output: NATIVE for a memory type, FILE for the file type, LE or BE for pre-existing little or big endian types. Must be used with -o (output file) and it is recommended that -d (dataset) is used. B is an optional argument, defaults to NATIVE 
  * **Q** - is the sort index type. It can be "creation_order" or "name" (default) 
  * **Z** - is the sort order type. It can be "descending" or "ascending" (default)


##  Usage Examples
  * 1) Attribute foo of the group /bar_none in file quux.h5 ```
 h5dump --attribute=/bar_none/foo quux.h5

```



  * 2) Attribute "high/low" of the group /bar_none in the file quux.h5 ```
 h5dump --attribute="/bar_none/high\/low" quux.h5

```



  * 3) Selecting a subset from dataset /foo in file quux.h5 ```
 h5dump --dataset=/foo --start="0,1" --stride="1,1" --count="2,3" --block="2,2" quux.h5

```



  * 4) Saving dataset 'dset' in file quux.h5 to binary file 'out.bin' using a little-endian type ```
 h5dump --dataset=/dset --binary=LE --output=out.bin quux.h5

```



  * 5) Display two packed bits (bits 0-1 and bits 4-6) in the dataset /dset ```
 h5dump -d-dataset=/dset --packedbits=0,1,4,3 quux.h5

```



  * 6) Dataset foo in files file1.h5 file2.h5 file3.h5 ```
 h5dump --dataset=/foo file1.h5 file2.h5 file3.h5

```



  * 7) Dataset foo in split files splitfile-m.h5 splitfile-r.h5 ```
 h5dump --dataset=/foo --filedriver=split splitfile

```



  * 8) Dataset foo in multi files mf-s.h5, mf-b.h5, mf-r.h5, mf-g.h5, mf-l.h5 and mf-o.h5 ```
 h5dump --dataset=/foo --filedriver=multi mf

```



  * 9) Dataset foo in family files fam00000.h5 fam00001.h5 and fam00002.h5 ```
 h5dump --dataset=/foo --filedriver=family fam%05d.h5

```



Previous Chapter [h5diff](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_f__u_g.html#sec_cltools_h5diff) - Next Chapter [h5format_convert](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__f_c__u_g.html#sec_cltools_h5format_convert)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
###  XML Deprecated
The XML option for h5dump has been deprecated. The methods for displaying XML output has not been updated since 1.10.0 was released. Many new features introduced into the library have never been added to the XML generation functions. Also the dtd and xsd files for XML have been archived, and copies of the files, HDF5-File.dtd and HDF5-File.xsd, have been saved in the "source"/tools/test/h5dump/testfiles/xml folder. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


