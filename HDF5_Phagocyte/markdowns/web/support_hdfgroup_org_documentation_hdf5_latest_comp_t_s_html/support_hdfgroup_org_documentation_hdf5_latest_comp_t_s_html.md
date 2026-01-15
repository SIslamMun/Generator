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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title0 "H1")
  * [ An Overview of HDF5 Compression Troubleshooting](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title1 "H1")
  * [ If a Filter Was Not Applied](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title2 "H1")
  * [ How the HDF5 Library Configuration May Miss a Compression Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title3 "H2")
  * [ How Does the HDF5 Library Behave in the Absence of a Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title4 "H2")
  * [ How to Determine if the HDF5 Library was Configured with a Needed Compression Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title5 "H2")
  * [ How to Use HDF5 Tools to Investigate Missing Compression Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title6 "H2")
  * [ Example Program](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title7 "H2")
  * [ If a Compression Filter Was Not Effective](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title8 "H1")
  * [ Compression Comparisons](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title9 "H2")
  * [ An Alternative to Compression](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title10 "H2")
  * [ Other Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#title11 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Compression Troubleshooting
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
The purpose of this technical note is to help HDF5 users with troubleshooting problems with [HDF5 Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html), especially with compression filters. The document assumes that the reader knows HDF5 basics and is aware of the compression feature in HDF5.
#  Introduction
One of the most powerful features of HDF5 is the ability to modify, or “filter,” data during I/O. Filters provided by the HDF5 Library, “predefined filters”, include several types of data compression, data shuffling and checksum. Users can implement their own “user-defined filters” and use them with the HDF5 Library.
By far the most common user-defined filters are ones that perform data compression. While the programming model and usage of the compression filters are straightforward, it is easy, especially for novice users, to overlook important details when implementing compression filters and to end up with data that is not modified as they would expect.
The purpose of this document is to describe how to diagnose situations where the data in a file is not compressed as expected.
#  An Overview of HDF5 Compression Troubleshooting
Sometimes users may find that HDF5 data was not compressed in a file or that the compression ratio is very small. By themselves, these results do not mean that compression did not work or did not work well. These results suggest that something might have gone wrong when a compression filter was applied. How can users determine the true cause of the problem?
There are two major reasons why a filter did not produce the desired result: it was not applied, or it was not effective.
#### The filter was not applied
If a filter was not applied at all, then it was not included at compile time when the library was built or was not found at run time for dynamically loaded filters.
The absence or presence of HDF5 predefined filters can be confirmed by examining the installed HDF5 files or by using HDF5 API calls. The absence or presence of all filter types can be confirmed by running HDF5 command-line tools on the produced HDF5 files. See [If a Filter Was Not Applied](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#sec_compts_notapp) for more information.
#### The filter was applied but was not effective
The effectiveness of compression filters is a complex matter and is only briefly covered this document. See [If a Compression Filter Was Not Effective](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#sec_compts_ineff) for more information. This section gives a short overview of the problem and provides an example in which the advantages of different compression filters and their combinations are shown.
#  If a Filter Was Not Applied
This section discusses how it may happen that a compression filter is not available to an application and describes the behavior of the HDF5 Library in the absence of the filter. Then we walk through how to troubleshoot the problem by checking the HDF5 installation, by examining what an application can do at run time to see if a filter is available, and by using some HDF5 command line tools to see if a filter was applied.
Note that there are internal predefined filters: 
[H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce) | The gzip compression, or deflation, filter   
---|---  
[H5Z_FILTER_SZIP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a421d9941c68ebb776573baeb9aa77cd2) | The SZIP compressionfilter   
[H5Z_FILTER_NBIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a8cc463fa1979bd4bfa0dd9aa6a41e49d) | The N-bit compression filter   
[H5Z_FILTER_SCALEOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a745d2ccb4f7712ed78ef5e562e27d2ca) | The scale-offset compression filter   
[H5Z_FILTER_SHUFFLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#aa723f1a71601bf22c95620a490ecf1af) | The shuffle algorithm filter   
[H5Z_FILTER_FLETCHER32](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a59ca894c9c2b99b1614b0c46a7407f1c) | The Fletcher32 checksum, or error checking, filter   
These are enabled by default by both **configure** and **CMake** builds. While these filters can be disabled intentionally with the **configure** flag `–disablefilters`, disabling them is not recommended. The discussion and the examples in this document focus on compression filters, but everything said can be applied to other missing internal filters as well.
##  How the HDF5 Library Configuration May Miss a Compression Filter
The HDF5 Library uses external libraries for data compression. The two predefined compression methods are **gzip3** and **szip** or **libaec** , and these can be requested at the HDF5 Library configuration time (compile time). User-defined compression filters and the corresponding libraries are usually linked with an application or provided as a dynamically loaded library.
**Note** that the **libaec** library is a replacement for the original **szip** library. The **libaec** library is a freely available, open-source library that provides compression and decompression functionality and is compatible with the **szip** filter. The **libaec** library can be used as a drop-in replacement for the **szip** , but requires two libraries to be present on the system: **libaec.a(so,dylib,lib)** and **libsz.a(so,dylib,lib)**. Everywhere in this document, the term **szip** refers to the **szip** filter and the **libaec** library.
**gzip** and **szip** require the **libz.a(so,dylib,lib)** and **libsz.a(so,dylib,lib)/libaec**.a(so,dylib,lib) libraries, respectively, to be present on the system and to be enabled during HDF5 configuration with this CMake configure command: 
cmake -G "Unix Makefiles" -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -D<other flags>
With CMake both libraries have to be explicitly enabled. The source code distribution’s `config/cmake/cacheinit.cmake` file will enable both filters along with setting other options. Users can overwrite the defaults by using -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF -DHDF5_ENABLE_ZLIB_SUPPORT:BOOL=OFF with the “cmake –C” command. See the [INSTALL_CMake.txt](https://github.com/HDFGroup/hdf5/blob/develop/release_docs/INSTALL_CMake.txt) file under the release_docs directory in the HDF5 source distribution.
If compression is not requested or found at configuration time, the compression method is not registered with the library and cannot be applied when data is written or read. For example, the [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) tool will not be able to remove an **szip** compression filter from a dataset if the **szip** library was not configured into the library against which the tool was built. The next section discusses the behavior of the HDF5 Library in the absence of filters.
##  How Does the HDF5 Library Behave in the Absence of a Filter
By design, the HDF5 Library allows applications to create and write datasets using filters that are not available at creation/write time. This feature makes it possible to create HDF5 files on one system and to write data on another system where the HDF5 Library is configured with or without the requested filter.
Let’ s recall the HDF5 programming model for enabling filters.
An HDF5 application uses one or more `H5Pset_<filter>` calls to configure a dataset’s filter pipeline at its creation time. The excerpt below shows how a **gzip** filter is added to a pipeline5 with [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level."). 
// Create the dataset creation property list, add the gzip
// compression filter and set the chunk size.
dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
status = [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d) (dcpl, 9);
status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b) (dcpl, 2, chunk);
dset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, DATASET,…, dcpl,…);
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)
herr_t H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t dim[])
Sets the size of the chunks used to store a chunked layout dataset.
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)
herr_t H5Pset_deflate(hid_t plist_id, unsigned level)
Sets deflate (GNU gzip) compression method and compression level.
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
For all internal filters (**shuffle** , **fletcher32** , **scaleoffset** , and **nbit**) and the external **gzip** filter, the HDF5 Library does not check to see if the filter is registered when the corresponding `H5Pset_<filter>` function is called. The only exception to this rule is [H5Pset_szip](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga37de4b6071a94574cfab5cd6de9c3fc6 "Sets up use of the SZIP compression filter.") which will fail if szip was not configured in or is configured with a decoder only. Hence, in the example above, [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level.") will succeed. The specified filter will be added to the dataset’s filter pipeline and will be applied to any data written to this dataset.
When `H5Pset_<filter>` is called, a record for the filter is added to the dataset’s object header in the file, and information about the filter can be queried with the HDF5 APIs and displayed by HDF5 tools such as [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump). The presence of filter information in a dataset’s header does not mean that the filter was actually applied to the dataset’s data, as will be explained later in this document. See [How to Use HDF5 Tools to Investigate Missing Compression Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#subsec_compts_notapp_tools) for more information on how to use [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) and [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug) to determine if the filter was actually applied.
The success of further write operations to a dataset when filters are missing depends on the filter type.
By design, an HDF5 filter can be optional or required. This filter mode defines the behavior of the HDF5 Library during write operations. In the absence of an optional filter, [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") calls will succeed and data will be written to the file, bypassing the filter. A missing required filter will cause [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") calls to fail. Clearly, [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") calls will fail when filters that are needed to decode the data are missing.
The HDF5 Library has only one required internal filter, **Fletcher32** (checksum creation), and one required external filter, **szip**. As mentioned earlier, only the **szip** compression ([H5Pset_szip](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga37de4b6071a94574cfab5cd6de9c3fc6 "Sets up use of the SZIP compression filter.")) will flag the absence of the filter. If, despite the missing filter, an application goes on to create a dataset via [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), the call will succeed, but the **szip** filter will not be added to the filter pipeline. This behavior is different from all other filters that may not be present, but will be added to the filter pipeline and applied during I/O. See the [Using HDF5 APIs](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#subsubsec_compts_notapp_need_api) section for more information on how to determine if a filter is available and to avoid writing data while the filter is missing.
Developers who create their own filters should use the **flags** parameter in [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") to declare if the filter is optional or required. The filter type can be determined by calling [H5Pget_filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15) and checking the value of the **flags** parameter.
For more information on filter behavior in HDF5, see [HDF5 Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html).
##  How to Determine if the HDF5 Library was Configured with a Needed Compression Filter
The previous section described how the HDF5 Library could be configured without certain compression filters and the resulting expected library behavior.
The following subsections explain how to determine if a compression method is configured in the HDF5 Library and how to avoid accessing data if the filter is missing.
###  Examine the hdf5lib.settings File
To see how the library was configured and built, users should examine the `hdf5lib.settings` text file found in the lib directory of the HDF5 installation point and search for the lines that contain the `“I/O filters”` string. The `hdf5lib.settings` file is automatically generated at configuration time when the HDF5 Library is built with **configure** on Unix or with **CMake** on Unix and Windows, and it should contain the following lines: 
I/O filters (external): deflate(zlib),szip(encoder)
I/O filters (internal): shuffle,fletcher32,nbit,scaleoffset
The same lines in the file generated by **CMake** look slightly different: 
I/O filters (external): DEFLATE ENCODE DECODE
I/O filters (internal): SHUFFLE FLETCHER32 NBIT SCALEOFFSET
`“ENCODE DECODE”` indicates that both the **szip** compression encoder and decoder are present. This inconsistency between configure and CMake generated files will be removed in a future release. These lines show the compression libraries configured with HDF5. Here is an example of the same output when external compression filters are absent: 
I/O filters (external):
I/O filters (internal): shuffle,fletcher32,nbit,scaleoffset
Depending on the values listed on the I/O filters (external) line, users will be able to tell if their HDF5 files are compressed appropriately. If **szip** is not included in the build, data files will not be compressed with **szip**. If **gzip** is not included in the build and is not installed on the system, then data files will not be compressed with **gzip**.
If the `hdf5lib.settings` file is not present on the system, then users can examine a public header file or the library binary file to find out if a filter is present, as is discussed in the next two sections.
###  Examine the H5pubconf.h Header File
To see if a filter is present, users can also inspect the HDF5 public header file installed under the include directory of the HDF5 installation point. If the compression and internal filters are present, the corresponding symbols will be defined as follows: 
// Define if support for deflate (zlib) filter is enabled
#define H5_HAVE_FILTER_DEFLATE 1
// Define if support for Fletcher32 checksum is enabled
#define H5_HAVE_FILTER_FLETCHER32 1
// Define if support for nbit filter is enabled
#define H5_HAVE_FILTER_NBIT 1
// Define if support for scaleoffset filter is enabled
#define H5_HAVE_FILTER_SCALEOFFSET 1
// Define if support for shuffle filter is enabled
#define H5_HAVE_FILTER_SHUFFLE 1
// Define if support for Szip filter is enabled
#define H5_HAVE_FILTER_SZIP 1
If a compression or internal filter was not configured, the corresponding lines will be commented out as follows: 
// Define if support for deflate (zlib) filter is enabled
// #undef H5_HAVE_FILTER_DEFLATE 
###  Check the HDF5 Library’s Binary
The HDF5 Library’s binary contains summary output similar to what is stored in the `hdf5lib.settings` file. Users can run the Unix `“strings”` command to get information about the configured filters: 
% strings libhdf5.a(so) | grep "I/O filters ("
I/O filters (external): deflate(zlib),szip(encoder)
I/O filters (internal): shuffle,fletcher32,nbit,scaleoffset
When compression filters are not configured, the output of the command above will be: 
I/O filters (external):
I/O filters (internal): shuffle,fletcher32,nbit,scaleoffset
On Windows one can use the `dumpbin /all` command, and then view and search the output for strings like `DEFLATE`, `FLETCHER32`, `DECODE`, and `ENCODE`. 
…..
10201860: 4E 0A 20 20 20 20 20 20 20 20 20 49 2F 4F 20 66 N. I/O f
10201870: 69 6C 74 65 72 73 20 28 65 78 74 65 72 6E 61 6C ilters (external
10201880: 29 3A 20 20 44 45 46 4C 41 54 45 20 44 45 43 4F ): DEFLATE DECO
10201890: 44 45 20 45 4E 43 4F 44 45 0A 20 20 20 20 20 20 DE ENCODE.
102018A0: 20 20 20 49 2F 4F 20 66 69 6C 74 65 72 73 20 28 I/O filters (
102018B0: 69 6E 74 65 72 6E 61 6C 29 3A 20 20 53 48 55 46 internal): SHUF
102018C0: 46 4C 45 20 46 4C 45 54 43 48 45 52 33 32 20 4E FLE FLETCHER32 N
102018D0: 42 49 54 20 53 43 41 4C 45 4F 46 46 53 45 54 0A BIT SCALEOFFSET.
###  Check the Compiler Script
Developers can also use the compiler scripts such as `h5cc` to verify that a compression library is present and configured in. Use the `- show` option with any of the compilers scripts found in the bin subdirectory of the HDF5 installation directory. The presence of `–lsz` and `–lz` options among the linker flags will confirm that **szip** or **gzip** were compiled with the HDF5 Library. See the sample below 
$ h5cc -show
gcc -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -L/mnt/hdf/packages/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/v1812/Linux64_2.6/standard/lib
/mnt/hdf/packages/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/v1812/Linux64_2.6/standard/lib/libhdf5_hl.a
/mnt/hdf/packages/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/v1812/Linux64_2.6/standard/lib/libhdf5.a -lsz -lz -lrt
-ldl -lm -Wl,-rpath -Wl,/mnt/hdf/packages/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/v1812/Linux64_2.6/standard/lib
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26
###  Examine the hdf5-config.cmake File
**CMake** users can check the `hdf5-config.cmake` file in the **CMake** installation directory. The file will indicate what options were used to configure the HDF5 Library. The variables in the "User Options" section can be used by developers programmatically to determine if a filter was configured in.
After using `find-package(HDF5)` **CMake** can test the setting of these variables as shown below: 
find_package (HDF5 NAMES [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html) COMPONENTS C)
if (HDF5_ENABLE_ZLIB_SUPPORT)
message(STATUS "gzip filter is available.")
else()
message(STATUS "gzip filter is not available.")
endif()
###  Using HDF5 APIs
Applications can check filter availability at run time. In order to check the filter’s availability with the HDF5 Library, users should know the filter identifier (for example, [H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce)) and call the [H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee "Determines whether a filter is available.") function as shown in the example below. Use [H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba "Retrieves information about a filter.") to determine if the filter is configured to decode data, to encode data, neither, or both. 
// Check if gzip compression is available and can be used for both compression and decompression.
avail = [H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee)([H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce));
if (!avail) {
printf ("gzip filter not available.\n");
return 1;
}
status = [H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba) ([H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce), &filter_info);
if ( !(filter_info & [H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a)) ||
!(filter_info & [H5Z_FILTER_CONFIG_DECODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a4dead61ceb139a3f97505d6e52eb1b8a)) ) {
printf ("gzip filter not available for encoding and decoding.\n");
return 1;
}
[H5Z_FILTER_CONFIG_DECODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a4dead61ceb139a3f97505d6e52eb1b8a)
#define H5Z_FILTER_CONFIG_DECODE_ENABLED
**Definition** H5Zpublic.h:273
[H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce)
#define H5Z_FILTER_DEFLATE
**Definition** H5Zpublic.h:66
[H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a)
#define H5Z_FILTER_CONFIG_ENCODE_ENABLED
**Definition** H5Zpublic.h:270
[H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee)
htri_t H5Zfilter_avail(H5Z_filter_t id)
Determines whether a filter is available.
[H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba)
herr_t H5Zget_filter_info(H5Z_filter_t filter, unsigned int *filter_config_flags)
Retrieves information about a filter.
[H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee "Determines whether a filter is available.") can be used to find filters that are registered with the library or are available via dynamically loaded libraries. For more information, see [Using Dynamically-Loadable Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsubsec_dataset_filters_dyn).
Currently there is no HDF5 API call to retrieve a list of all of the registered or dynamically loaded filters. The default installation directories for HDF5 dynamically loaded filters are `/usr/local/hdf5/lib/plugin` on Unix and `ALLUSERSPROFILE%\hdf5[](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)\lib\plugin` on Windows. Users can also check to see if the environment variable `HDF5_PLUGIN_PATH` is set on the system and refers to a directory with available plugins.
##  How to Use HDF5 Tools to Investigate Missing Compression Filters
In this section, we will use the [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump), [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls), and [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug) command-line utilities to see if a file was created with an HDF5 Library that did or did not have a compression filter configured in. For more information on these tools, see the [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html#sec_cltools) page in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html).
###  How to Use h5dump to Examine Files with Compressed Data
The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) command-line tool can be used to see if a file uses a compression filter. The tool has two flags that will limit the output: the `–p` flag causes dataset properties including compression filters to be displayed, and the `–H` flag is used to suppress the output of data. The program provided in the [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html#sec_cltools) section creates a file called `h5ex_d_gzip.h5`. The output of [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) shows that the **gzip** compression filter set to level 9 was added to the DS1 dataset filter pipeline at creation time.
$ [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/bin/h5dump -p -H *.h5
HDF5 "h5ex_d_gzip.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 32, 64 ) / ( 32, 64 ) }
STORAGE_LAYOUT {
CHUNKED ( 5, 9 )
SIZE 5018 (1.633:1 COMPRESSION)
}
FILTERS {
COMPRESSION DEFLATE { LEVEL 9 }
}
FILLVALUE {
FILL_TIME [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
VALUE 0
}
ALLOCATION_TIME {
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
}
}
}
}
[H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
@ H5D_FILL_TIME_IFSET
**Definition** H5Dpublic.h:119
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
@ H5D_ALLOC_TIME_INCR
**Definition** H5Dpublic.h:94
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
#define H5T_STD_I32LE
**Definition** H5Tpublic.h:471
The output also shows a compression ratio defined as (original size)/(storage size). The size of the stored data is 5018 bytes vs. 8192 bytes of uncompressed data, a ratio of 1.663. This shows that the filter was successfully applied.
Now let’s look at what happens when the same program is linked against an HDF5 Library that was not configured with the **gzip** library.
Notice that some chunks are only partially filled. 56 chunks (7 along the first dimension and 8 along the second dimension) are required to store the data. Since no compression was applied, each chunk has size `5x9x4 = 180` bytes, resulting in a total storage size of 10,080 bytes. With an original size of 8192 bytes, the compression ratio is 0.813 (in other words, less than 1) and visible in the output below. 
$ [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/bin/h5dump -p -H *.h5
HDF5 "h5ex_d_gzip.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 32, 64 ) / ( 32, 64 ) }
STORAGE_LAYOUT {
CHUNKED ( 5, 9 )
SIZE 10080 (0.813:1 COMPRESSION)
}
FILTERS {
COMPRESSION DEFLATE { LEVEL 9 }
}
FILLVALUE {
FILL_TIME [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
VALUE 0
}
ALLOCATION_TIME {
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
}
}
}
}
As discussed in the [How Does the HDF5 Library Behave in the Absence of a Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#subsec_compts_notapp_behave), the presence of a filter in an object’s filter pipeline **does not imply** that it will be applied unconditionally when data is written.
If the compression ratio is less than 1, compression is not applied. If it is 1, and compression is shown by [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump), more investigation is needed; this will be discussed in the next section.
###  How to Use h5ls and h5debug to Find a Missing Compression Filter
Filters operate on chunked datasets. A filter may be ineffective for one chunk (for example, the compressed data is bigger than the original data), and succeed on another. How can users discern if a filter is missing or just ineffective (and as a result non-compressed data was written)? The [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) and [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug) command-line tools can be used to investigate the issue.
First, let’s take a look at what kind of information [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) displays about the dataset DS1 in our example file, which was written with an HDF5 library that has the **deflate** filter configured in: 
$ h5ls -vr h5ex_d_gzip.h5
Opened "h5ex_d_gzip.h5" with sec2 driver.
/ Group
Location: 1:96
Links: 1
/DS1 Dataset {32/32, 64/64}
Location: 1:800
Links: 1
Chunks: {5, 9} 180 bytes
Storage: 8192 logical bytes, 5018 allocated bytes, 163.25% utilization
Filter-0: deflate-1 OPT {9}
Type: native int
We see output similar to [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) output with the compression ratio at 163%.
Now let’s compare this output with another dataset **DS1** , but this time the dataset was written with a program linked against an HDF5 library without the **gzip** filter present. 
$ h5ls -vr h5ex_d_gzip.h5
Opened "h5ex_d_gzip.h5" with sec2 driver.
/ Group
Location: 1:96
Links: 1
/DS1 Dataset {32/32, 64/64}
Location: 1:800
Links: 1
Chunks: {5, 9} 180 bytes
Storage: 8192 logical bytes, 10080 allocated bytes, 81.27% utilization
Filter-0: deflate-1 OPT {9}
Type: native int
The [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) output above shows that the **gzip** filter was added to the filter pipeline of the dataset **DS1**. It also shows that the compression ratio is less than 1. We can confirm by using [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug) that the filter was not applied at all, and, as a result of the missing filter, the individual chunks were not compressed.
From the [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) output we know that the dataset object header is located at address 800. We retrieve the dataset object header at address 800 and search the layout message for the address of the chunk index B-tree as shown in the excerpt of the [h5debug](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_g__u_g.html#sec_cltools_h5debug) output below: 
$ h5debug h5ex_d_gzip.h5 800
Reading signature at address 800 (rel)
Object Header...
…..
Message 4...
Message ID (sequence number): 0x0008 `layout' (0)
Dirty: FALSE
Message flags: <C>
Chunk number: 0
Raw message data (offset, size) in chunk: (144, 24) bytes
Message Information:
Version: 3
Type: Chunked
Number of dimensions: 3
Size: {5, 9, 4}
Index Type: v1 B-tree
B-tree address: 1400
Now we can retrieve the B-tree information: 
$ h5debug h5ex_d_gzip.h5 1400 3
Reading signature at address 1400 (rel)
Tree type ID: H5B_CHUNK_ID
Size of node: 2616
Size of raw (disk) key: 32
Dirty flag: False
Level: 0
Address of left sibling: UNDEF
Address of right sibling: UNDEF
Number of children (max): 56 (64)
Child 0...
Address: 4016
Left Key:
Chunk size: 180 bytes
Filter mask: 0x00000001
Logical offset: {0, 0, 0}
Right Key:
Chunk size: 180 bytes
Filter mask: 0x00000001
Logical offset: {0, 9, 0}
Child 1...
Address: 4196
Left Key:
Chunk size: 180 bytes
  * Users have to supply the chunk rank. According to the HDF5 [File Format](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html#sec_spec_ff) Specification, this is the dataset rank plus 1; in other words, 3.


We see that the size of each chunk is 180 bytes: in other words, compression was not successful. The filter mask value 0x00000001 indicates that filter was not applied. For more information on the filter mask, see the [III.A.1. Disk Format: Level 1A1 - Version 1 B-trees](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html#subsubsec_fmt3_infra_btrees_v1) section in the HDF5 [File Format](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html#sec_spec_ff) Specification.
##  Example Program
The example program used to create the file discussed in this document is a modified version of the program available at [h5ex_d_gzip.c](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5D/h5ex_d_gzip.c). It was modified to have chunk dimensions not be factors of the dataset dimensions. Chunk dimensions were chosen for demonstration purposes only and are not recommended for real applications. 
#include <stdlib.h>
#define FILE "h5ex_d_gzip.h5"
#define DATASET "DS1"
#define DIM0 32
#define DIM1 64
#define CHUNK0 5
#define CHUNK1 9
int main (void)
{
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, space, dset, dcpl; // Handles
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status;
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) avail;
[H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) filter_type;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims[2] = {DIM0, DIM1}, chunk[2] = {CHUNK0, CHUNK1};
size_t nelmts;
unsigned int flags, filter_info;
int wdata[DIM0][DIM1], // Write buffer
rdata[DIM0][DIM1], // Read buffer
max, i, j;
// Initialize data.
for (i=0; i<DIM0; i++)
for (j=0; j<DIM1; j++)
wdata[i][j] = i * j - j;
// Create a new file using the default properties.
file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) (FILE, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Create dataspace. Setting maximum size to NULL sets the maximum
// size to be the current size.
space = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae) (2, dims, NULL);
// Create the dataset creation property list, add the gzip
// compression filter and set the chunk size.
dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
status = [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d) (dcpl, 9);
status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b) (dcpl, 2, chunk);
// Create the dataset.
dset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, DATASET, [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Write the data to the dataset.
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), wdata[0]);
// Close and release resources.
status = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (dcpl);
status = [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609) (dset);
status = [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1) (space);
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) (file);
return 0;
}
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237)
int H5Z_filter_t
Filter identifiers.
**Definition** H5Zpublic.h:55
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)
int htri_t
**Definition** H5public.h:282
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)
herr_t H5Sclose(hid_t space_id)
Releases and terminates access to a dataspace.
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
#  If a Compression Filter Was Not Effective
There is no “one size fits all” compression filter solution. Users have to consider a number of characteristics such as the type of data, the desired compression ratio, the encoding/decoding speed, the general availability of a compression filter, and licensing among other issues before committing to a compression filter. This is especially true for data producers. The way data is written will affect how much bandwidth consumers will need to download data products, how much system memory and time will be required to read the data, and how many data products can be stored on the users’ system to name a few issues. Users should plan on experimenting with various compression filters and settings. The following are some suggestions for where to start finding the best compression filter for your data: 
  * Find the compression filter which operates best on the type of data in the file and for the objectives of the file users. Different applications, data providers, and data consumers will work differently with different compression filters. For example, **szip** compression is fast but typically achieves smaller compression ratios for floating point data than gzip. For more information on **szip** , see the [Using the SZip Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsubsec_dataset_filters_szip) page. 
  * Once you have the right compression method, find the right parameters. For example, **gzip** compression at level 6 usually achieves a compression ratio comparable to level 9 in less time. For more information on compression levels, see the [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level.") entry on the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) page of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html). 
  * Data preprocessing using a filter such as **shuffling** in combination with compression can drastically improve the compression ratio. See the [Compression Comparisons](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#subsec_compts_ineff_comp) section below for more information.


The [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) tool can be used to experiment with the data to address the items above.
Users should also look beyond compression. An HDF5 file may contain a substantial amount of unused space. The [h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#sec_cltools_h5stat) tool can be used to determine if space is used efficiently in an HDF5 file, and the [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) tool can be used to reduce the amount of unused space in an HDF5 file. See [An Alternative to Compression](https://support.hdfgroup.org/documentation/hdf5/latest/_comp_t_s.html#subsec_compts_ineff_alt) for more information.
##  Compression Comparisons
An extensive comparison of different compression filters is outside the scope of this document. However, it is easy to show that, unless a suitable compression method or an advantageous filter combination is chosen, applying the same compression filter to different types of data may not reduce HDF5 file size as much as possible.
For example, we looked at a NASA weather data product file packaged with it its geolocation information (the file name is `GCRIOREDRO_npp_d20030125_t0702533_e0711257_b00993_c20140501163427060570_XXXX_XXX.h5`) and used [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) to apply three different compressions to the original file: 
  * gzip with compression level 7 
  * szip compression using NN mode and 32-bit block size 
  * Shuffle in combination with gzip compression level 7


Then we compared the sizes of the 32-bit floating dataset `/All_Data/CrIMSS-EDR-GEOTC_All/Height` when different types of compression were used and compared for the sizes of the 32-bit integer dataset `/All_Data/CrIMSS-EDR_All/FORnum`. The results are shown in the table below. 
Table 1: Compression ratio for different types of compressions when using h5repack Data  | Original  | gzip Level 7  | szip Using NN Mode and Blocksize 32  | Shuffle and gzip Level 7   
---|---|---|---|---  
32-bit Floats  | 1  | 2.087  | 1.628  | 2.56   
32-bit Integers  | 1  | 3.642  | 10.832  | 38.20   
The combination of the **shuffle** filter and **gzip** compression level 7 worked well on both floating point and integer datasets, as shown in the fifth column of the table above. **gzip** compression worked better than **szip** on the floating point dataset, but not on the integer dataset as shown by the results in columns three and four. Clearly, if the objective is to minimize the size of the file, datasets with different types of data have to be compressed with different compression methods.
For more information on the **shuffle** filter, see the [Data Pipeline Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsubsec_dataset_transfer_filter) section in the [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset) chapter of the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html). See also the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) for the [H5Pset_shuffle](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga31e09cb0bf2da2893eed8a72220e6521 "Sets up use of the shuffle filter.") function call entry.
##  An Alternative to Compression
Sometimes HDF5 files contain unused space. The [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) command-line tool can be used to reduce the amount of unused space in a file without changing any storage parameters of the data. For example, running [h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#sec_cltools_h5stat) on the file `GCRIOREDRO_npp_d20030125_t0702533_e0711257_b00993_c20140501163524579819_XXXX_XXX.h5` shows: 
Summary of file space information:
File metadata: 425632 bytes
Raw data: 328202 bytes
Unaccounted space: 449322 bytes
Total space: 1203156 bytes
After running [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack), the file shows a 10-fold reduction in unaccounted space: 
Summary of file space information:
File metadata: 425176 bytes
Raw data: 328202 bytes
Unaccounted space: 45846 bytes
Total space: 799224 bytes
There is also a small reduction in file metadata space. For more information on [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack) and [h5stat](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__s_t__u_g.html#sec_cltools_h5stat), see the [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html#sec_cltools) page in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html).
#  Other Resources
See the following documents published by The HDF Group for more information. 
  * See the [Creating a Compressed Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_com_dset.html#secLBComDsetCreate) tutorial. 
  * The “Filter Behavior in HDF5” note is part of the [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") function call entry. See the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


