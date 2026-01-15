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
  * [ h5import](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title1 "H2")
  * [ Description](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title2 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title3 "H2")
  * [ Help](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title4 "H2")
  * [ Configuration File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title5 "H2")
  * [ Usage Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5import Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5import
##  Introduction
With h5import, you can convert data stored in one or more ASCII or binary files into one or more datasets (in accordance with the user-specified type and storage properties) in an existing or new HDF5 file.
##  Description
The primary objective of the utility is to convert floating point or integer data stored in **ASCII** text or binary form into a dataset according to the type and storage properties specified by the user. The utility can also accept **ASCII** text files and store the contents in a compact form as an array of one-dimensional strings.
The input data to be written as a dataset can be provided to the utility in one of the following forms: 
  * 1. ASCII text file with numeric data (floating point or integer data) 
  * 2. Binary file with native floating point data (32-bit or 64-bit) 
  * 3. Binary file with native integer (signed or unsigned) data (8-bit or 16-bit or 32-bit or 64-bit) 
  * 4. ASCII text file containing strings (text data)


Every input file is associated with a configuration file also provided as an input to the utility. (See Section [Configuration File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#subsec_cltools_h5import_config) to know how it is to be organized). The class, size and dimensions of the input data is specified in this configuration file. A point to note is that the floating point data in the **ASCII** text file may be organized in the fixed floating form (for example 323.56) or in scientific notation (for example 3.23E+02). A different input-class specification is to be used for both forms.
The utility extracts the input data from the input file according to the specified parameters and saves it into an HDF5 dataset.
The user can specify output type and storage properties in the configuration file. The user is required to specify the path of the dataset. If the groups in the path leading to the dataset do not exist, the groups will be created by the utility. If no group is specified, the dataset will be created under the root group.
In addition to the name, the user is also required to provide the class and size of output data to be written to the dataset and may optionally specify the output-architecture, and the output-byte-order. If output-architecture is not specified, the default is **NATIVE**. Output-byte-orders are fixed for some architectures and may be specified only if output- architecture is **IEEE** , **UNIX** or **STD**.
Also, layout and other storage properties such as compression, external storage and extendible datasets may be optionally specified. The layout and storage properties denote how raw data is to be organized on the disk. If these options are not specified, the default is **Contiguous** layout and storage.
The dataset can be organized in any of the following ways: 
  * 1. **Contiguous**
  * 2. **Chunked**
  * 3. **External Storage File** (has to be contiguous) 
  * 4. **Extendible data sets**(has to be chunked) 
  * 5. **Compressed** (has to be chunked) 
  * 6. **Compressed & Extendible** (has to be chunked)


If the user wants to store raw data in a non-HDF5 file then the external storage file option is to be used and the name of the file is to be specified.
If the user wants the dimensions of the dataset to be unlimited, the extendible data set option can be chosen.
The user may also specify the type of compression and the level to which the data set must be compressed by setting the compressed option.
##  Usage
#### h5import -h[elp], OR h5import <infile> -c[onfig] <configfile> [<infile> -c[config]<confile2>...] -o[utfile] <outfile>
##  Help
  * **-h[elp]** Print a usage message and exit


###  Program Options
  * **< infile(s)>** Name of the [Input](https://support.hdfgroup.org/documentation/hdf5/latest/struct_input.html) file(s), containing a single n-dimensional floating point or integer array in either ASCII text, native floating point(32-bit or 64-bit) or native integer(8-bit or 16-bit or 32-bit or 64-bit). Data to be specified in the order of fastest changing dimensions first.


  * **-c[config] <configfile>** Every input file should be associated with a configuration file and this is done by the -c option. <configfile> is the name of the configuration file. (See Section [Configuration File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#subsec_cltools_h5import_config)).


  * **-o[utfile] <outfile>** Name of the HDF5 output file. Data from one or more input files are stored as one or more data sets in <outfile>. The output file may be an existing file or it may be new, in which case it will be created.


##  Configuration File
The configuration file is an ASCII text file and must be the ddl formatted file (without data values) produced by **h5dump** when used with the options 
-o outfilename -b 
of a single dataset (-d) OR organized as **CONFIG-KEYWORD VALUE** pairs, one pair on each line.
The configuration file may have the following keywords each followed by an acceptable value.
###  Required KEYWORDS
  * **PATH**
  * **INPUT-CLASS**
  * **INPUT-SIZE**
  * **INPUT-BYTE-ORDER**
  * **RANK**
  * **DIMENSION-SIZES**
  * **OUTPUT-CLASS**
  * **OUTPUT-SIZE**


###  Optional KEYWORDS
  * **OUTPUT-ARCHITECTURE**
  * **OUTPUT-BYTE-ORDER**
  * **CHUNKED-DIMENSION-SIZES**
  * **COMPRESSION-TYPE**
  * **COMPRESSION-PARAM**
  * **EXTERNAL-STORAGE**
  * **MAXIMUM-DIMENSIONS**


###  Values for keywords
  * **PATH** Strings separated by spaces to represent the path of the dataset. If the groups in the path do not exist, they will be created. For example, 
    * PATH grp1/grp2/dataset1 
    * PATH: keyword 
    * grp1: group under the root. If non-existent will be created 
    * grp2: group under grp1. If non-existent will be created under grp1 
    * dataset1: the name of the dataset to be created


  * **INPUT-CLASS** String denoting the type of input data. 
    * TEXTIN 
    * TEXTFP 
    * FP 
    * IN 
    * STR 
    * TEXTUIN 
    * UIN
**INPUT-CLASS** "TEXTIN" denotes an ASCII text file with signed integer data in ASCII form, **INPUT-CLASS** "TEXTUIN" denotes an ASCII text file with unsigned integer data in ASCII form, "TEXTFP" denotes an ASCII text file containing floating point data in the fixed notation (325.34),   
"FP" denotes a floating point binary file, "IN" denotes a signed integer binary file, "UIN" denotes an unsigned integer binary file, & "STR" denotes an ASCII text file the contents of which should be stored as a 1-D array of strings.  
If **INPUT-CLASS** is "STR", then **RANK** , **DIMENSION-SIZES** , **OUTPUT-CLASS** , **OUTPUT-SIZE** , **OUTPUT-ARCHITECTURE** and **OUTPUT-BYTE-ORDER** will be ignored.


  * **INPUT-SIZE** Integer denoting the size of the input data (8, 16, 32, 64). 
    * For floating point, **INPUT-SIZE** can be 32 or 64. 
    * For integers (signed and unsigned) **INPUT-SIZE** can be 8, 16, 32 or 64.


  * **RANK** Integer denoting the number of dimensions.


  * **DIMENSION-SIZES** Integers separated by spaces to denote the dimension sizes for the number of dimensions determined by rank.


  * **OUTPUT-CLASS** String denoting data type of the dataset to be written ("IN","FP", "UIN")


  * **OUTPUT-SIZE** Integer denoting the size of the data in the output dataset to be written. If **OUTPUT-CLASS** is "FP", **OUTPUT-SIZE** can be 32 or 64. If **OUTPUT-CLASS** is "IN" or "UIN", **OUTPUT-SIZE** can be 8, 16, 32 or 64.


  * **OUTPUT-ARCHITECTURE** **STRING** denoting the type of output architecture. Can accept the following values 
    * STD 
    * IEEE 
    * INTEL 
    * CRAY 
    * MIPS 
    * ALPHA 
    * NATIVE (default) 
    * UNIX


  * **OUTPUT-BYTE-ORDER** String denoting the output-byte-order. Ignored if the **OUTPUT-ARCHITECTURE** is not specified or if it is **IEEE** , **UNIX** or **STD**. Can accept the following values. 
    * BE (default) 
    * LE


  * **CHUNKED-DIMENSION-SIZES** Integers separated by spaces to denote the dimension sizes of the chunk for the number of dimensions determined by rank. Required field to denote that the dataset will be stored with chunked storage. If this field is absent the dataset will be stored with contiguous storage.


  * **COMPRESSION-TYPE** String denoting the type of compression to be used with the chunked storage. Requires the **CHUNKED-DIMENSION-SIZES** to be specified. The only currently supported compression method is **GZIP**. Will accept the following value 
    * GZIP


  * **COMPRESSION-PARAM** Integer used to denote compression level and this option is to be always specified when the **COMPRESSION-TYPE** option is specified. The values are applicable only to **GZIP** compression.  
Value 1-9: The level of Compression.  
1 will result in the fastest compression while 9 will result in the best compression ratio.  
The default level of compression is 6.


  * **EXTERNAL-STORAGE** String to denote the name of the non-HDF5 file to store data to. Cannot be used if **CHUNKED-DIMENSIONS** or **COMPRESSION-TYPE** or **EXTENDIBLE-DATASET** is specified. Value <external-filename>: the name of the external file as a string to be used.


  * **MAXIMUM-DIMENSIONS** Integers separated by spaces to denote the maximum dimension sizes of all the dimensions determined by rank. Requires the **CHUNKED-DIMENSION-SIZES** to be specified. A value of -1 for any dimension implies **UNLIMITED** **DIMENSION** size for that particular dimension.


##  Usage Examples
  * **Configuration File may look like**
[PATH](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#ab0139008fdda107456f13f837872b410) work h5 pkamat First-set
INPUT-CLASS TEXTFP
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5) 3
DIMENSION-SIZES 5 2 4
OUTPUT-CLASS FP
OUTPUT-SIZE 64
OUTPUT-ARCHITECTURE IEEE
OUTPUT-BYTE-ORDER LE
CHUNKED-DIMENSION-SIZES 2 2 2
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
[PATH](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#ab0139008fdda107456f13f837872b410)
#define PATH
**Definition** h5import.h:340
The above configuration will accept a floating point array (5 x 2 x 4) in an ASCII file with the rank and dimension sizes specified and will save it in a chunked dataset (of pattern 2 X 2 X 2) of 64-bit floating point in the little-endian order and IEEE architecture. The dataset will be stored at "/work/h5/pkamat/First-set"


  * **Another configuration could be**
[PATH](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#ab0139008fdda107456f13f837872b410) Second-set
INPUT-CLASS IN
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5) 5
DIMENSION-SIZES 6 3 5 2 4
OUTPUT-CLASS IN
OUTPUT-SIZE 32
CHUNKED-DIMENSION-SIZES 2 2 2 2 2
EXTENDIBLE-DATASET 1 3
COMPRESSION-TYPE GZIP
COMPRESSION-PARAM 7
The above configuration will accept an integer array (6 X 3 X 5 x 2 x 4) in a binary file with the rank and dimension sizes specified and will save it in a chunked dataset (of pattern 2 X 2 X 2 X 2 X 2) of 32-bit floating point in native format (as output-architecture is not specified). The first and the third dimension will be defined as unlimited. The dataset will be compressed using GZIP and a compression level of 7.  
The dataset will be stored at 
/Second-set 


Previous Chapter [h5format_convert](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__f_c__u_g.html#sec_cltools_h5format_convert) - Next Chapter [h5jam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#sec_cltools_h5jam)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


