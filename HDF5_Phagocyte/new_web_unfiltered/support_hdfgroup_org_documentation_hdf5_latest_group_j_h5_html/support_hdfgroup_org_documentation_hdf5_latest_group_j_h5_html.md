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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title1 "H2")
  * [ Variables](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title3 "H2")
  * [◆ getOpenIDCount()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title4 "H2")
  * [◆ getOpenIDs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title5 "H2")
  * [◆ H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title6 "H2")
  * [◆ H5close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title7 "H2")
  * [◆ H5error_off()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title8 "H2")
  * [◆ H5error_on()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title9 "H2")
  * [◆ H5export_attribute()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title10 "H2")
  * [◆ H5export_dataset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title11 "H2")
  * [◆ H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title12 "H2")
  * [◆ H5get_libversion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title13 "H2")
  * [◆ H5open()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title14 "H2")
  * [◆ H5set_free_list_limits()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title15 "H2")
  * [◆ loadH5Lib()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title16 "H2")
  * [Variable Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title17 "H2")
  * [◆ H5_LIBRARY_NAME_PROPERTY_KEY](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title18 "H2")
  * [◆ H5PATH_PROPERTY_KEY](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title19 "H2")
  * [◆ LIB_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#title20 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#func-members) | [Variables](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#var-members)
HDF5 Library Java Interface
## Detailed Description
This class is the Java interface for the HDF5 library.
This code is the called by Java programs to access the entry points of the HDF5 library. Each routine wraps a single HDF5 entry point, generally with the arguments and return codes analogous to the C interface. 

See also
     [H5](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1_h5.html), C-API      [HDF5 Library and Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html), User Guide 
##  Functions  
---  
static final int  |  [getOpenIDCount](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gadea4d5366be3ed84b465c30646bf55a8) ()  
static final Collection< Long > |  [getOpenIDs](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaa08367f25e20573271bc84ec380cec29) ()  
static synchronized native int  |  [H5check_version](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga74ff3a8c9f3c24b63af2c6d8f0c5b55c) (int majnum, int minnum, int relnum)  
static synchronized native int  |  [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gad81ce9193f5c63cef4d66ad48bf29049) () throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5error_off](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga5d32fbd2ef1197cd813a8fb04fad597b) ()  
static synchronized native void  |  [H5error_on](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaf0996c24dcb63216b942707d993e4f53) ()  
static synchronized native void  |  [H5export_attribute](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga7f1556bb4c53e46c2c43c8608390b9f0) (String file_export_name, long dataset_id, String attribute_name, int binary_order) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5export_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga9be84065bef5bcc2d6408fd2fb55ab57) (String file_export_name, long file_id, String object_path, int binary_order) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5garbage_collect](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gabd2f2b344e59ed550aa8423cac557797) () throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5get_libversion](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga9f90a4479f0296a4355f1f08aa88564f) (int[] libversion) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga833fad7925ae765a6100a050b8c24ad6) () throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5set_free_list_limits](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaae1529dd54abc5e191cb53b464e569f5) (int reg_global_lim, int reg_list_lim, int arr_global_lim, int arr_list_lim, int blk_global_lim, int blk_list_lim) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static void  |  [loadH5Lib](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaf9c0f205c18b7fbc0f651786181f83a8) ()  
##  Variables  
---  
static final String  |  [H5_LIBRARY_NAME_PROPERTY_KEY](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga43b7d41cf2b9f89ccc6650ccc86dd2e0) = "hdf.hdf5lib.H5.loadLibraryName"  
static final String  |  [H5PATH_PROPERTY_KEY](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga56bd6da6f5e5b5f06350022a4469699d) = "hdf.hdf5lib.H5.hdf5lib"  
static final int  |  [LIB_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaa6bbaf809e566319082311c2b9572489) [] = {2, 0, 1}  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gadea4d5366be3ed84b465c30646bf55a8)getOpenIDCount()
| static final int getOpenIDCount  | ( |  | ) |   
---|---|---|---|---  
static  
Get number of open IDs. 

Returns
    Returns a count of open IDs 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaa08367f25e20573271bc84ec380cec29)getOpenIDs()
| static final Collection< Long > getOpenIDs  | ( |  | ) |   
---|---|---|---|---  
static  
Get the open IDs 

Returns
    Returns a collection of open IDs 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga74ff3a8c9f3c24b63af2c6d8f0c5b55c)H5check_version()
| static synchronized native int H5check_version  | ( | int |  _majnum_ ,   
---|---|---|---  
|  | int |  _minnum_ ,   
|  | int |  _relnum_ )  
static  
H5check_version verifies that the arguments match the version numbers compiled into the library. 

Parameters
     majnum | The major version of the library.   
---|---  
minnum | The minor version of the library.   
relnum | The release number of the library.  

Returns
    a non-negative value if successful. Upon failure (when the versions do not match), this function causes the application to abort (i.e., crash)  
See C API function: [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5check_version(unsigned majnum, unsigned minnum, unsigned relnum)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gad81ce9193f5c63cef4d66ad48bf29049)H5close()
| static synchronized native int H5close  | ( |  | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---  
static  
H5close flushes all data to disk, closes all file identifiers, and cleans up all memory used by the library. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga5d32fbd2ef1197cd813a8fb04fad597b)H5error_off()
| static synchronized native int H5error_off  | ( |  | ) |   
---|---|---|---|---  
static  
Turn off error handling. By default, the C library prints the error stack of the HDF5 C library on stdout. This behavior may be disabled by calling [H5error_off()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga5d32fbd2ef1197cd813a8fb04fad597b). 

Returns
    a non-negative value if successful 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaf0996c24dcb63216b942707d993e4f53)H5error_on()
| static synchronized native void H5error_on  | ( |  | ) |   
---|---|---|---|---  
static  
Turn on error handling. By default, the C library prints the error stack of the HDF5 C library on stdout. This behavior may be re-enabled by calling [H5error_on()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaf0996c24dcb63216b942707d993e4f53). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga7f1556bb4c53e46c2c43c8608390b9f0)H5export_attribute()
| static synchronized native void H5export_attribute  | ( | String |  _file_export_name_ ,   
---|---|---|---  
|  | long |  _dataset_id_ ,   
|  | String |  _attribute_name_ ,   
|  | int |  _binary_order_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5export_attribute is a utility function to save data in a file. 

Parameters
     file_export_name | The file name to export data into.   
---|---  
dataset_id | The identifier of the dataset containing the attribute.   
attribute_name | The attribute to be exported.   
binary_order | 99 - export data as text. 1 - export data as binary Native Order. 2 - export data as binary Little Endian. 3 - export data as binary Big Endian. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga9be84065bef5bcc2d6408fd2fb55ab57)H5export_dataset()
| static synchronized native void H5export_dataset  | ( | String |  _file_export_name_ ,   
---|---|---|---  
|  | long |  _file_id_ ,   
|  | String |  _object_path_ ,   
|  | int |  _binary_order_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5export_dataset is a utility function to save data in a file. 

Parameters
     file_export_name | The file name to export data into.   
---|---  
file_id | The identifier of the HDF5 file containing the dataset.   
object_path | The full path of the dataset to be exported.   
binary_order | 99 - export data as text. 1 - export data as binary Native Order. 2 - export data as binary Little Endian. 3 - export data as binary Big Endian. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gabd2f2b344e59ed550aa8423cac557797)H5garbage_collect()
| static synchronized native int H5garbage_collect  | ( |  | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---  
static  
H5garbage_collect collects on all free-lists of all types. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga9f90a4479f0296a4355f1f08aa88564f)H5get_libversion()
| static synchronized native int H5get_libversion  | ( | int[] | _libversion_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5get_libversion retrieves the major, minor, and release numbers of the version of the HDF library which is linked to the application. 

Parameters
     libversion | The version information of the HDF library.  
---|--- ```
     libversion[0] = The major version of the library.
     libversion[1] = The minor version of the library.
     libversion[2] = The release number of the library.

```


Returns
    a non-negative value if successful, along with the version information. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga833fad7925ae765a6100a050b8c24ad6)H5open()
| static synchronized native int H5open  | ( |  | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---  
static  
H5open initialize the library. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaae1529dd54abc5e191cb53b464e569f5)H5set_free_list_limits()
| static synchronized native int H5set_free_list_limits  | ( | int |  _reg_global_lim_ ,   
---|---|---|---  
|  | int |  _reg_list_lim_ ,   
|  | int |  _arr_global_lim_ ,   
|  | int |  _arr_list_lim_ ,   
|  | int |  _blk_global_lim_ ,   
|  | int |  _blk_list_lim_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5set_free_list_limits Sets limits on the different kinds of free lists. Setting a value of -1 for a limit means no limit of that type. These limits are global for the entire library. Each "global" limit only applies to free lists of that type, so if an application sets a limit of 1 MB on each of the global lists, up to 3 MB of total storage might be allocated (1MB on each of regular, array and block type lists).
The settings for block free lists are duplicated to factory free lists. Factory free list limits cannot be set independently currently. 

Parameters
     reg_global_lim | The limit on all "regular" free list memory used   
---|---  
reg_list_lim | The limit on memory used in each "regular" free list   
arr_global_lim | The limit on all "array" free list memory used   
arr_list_lim | The limit on memory used in each "array" free list   
blk_global_lim | The limit on all "block" free list memory used   
blk_list_lim | The limit on memory used in each "block" free list  

Returns
    a non-negative value if successful, along with the version information. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaf9c0f205c18b7fbc0f651786181f83a8)loadH5Lib()
| static void loadH5Lib  | ( |  | ) |   
---|---|---|---|---  
static  
load native library 
## Variable Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga43b7d41cf2b9f89ccc6650ccc86dd2e0)H5_LIBRARY_NAME_PROPERTY_KEY
| final String H5_LIBRARY_NAME_PROPERTY_KEY = "hdf.hdf5lib.H5.loadLibraryName"  
---  
static  
add system property to load library by name from library path, via System.loadLibrary() 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#ga56bd6da6f5e5b5f06350022a4469699d)H5PATH_PROPERTY_KEY
| final String H5PATH_PROPERTY_KEY = "hdf.hdf5lib.H5.hdf5lib"  
---  
static  
add system property to load library by path 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html#gaa6bbaf809e566319082311c2b9572489)LIB_VERSION
| final int LIB_VERSION[] = {2, 0, 1}  
---  
static  
The version number of the HDF5 library: 
  * LIB_VERSION[0]: The major version of the library. 
  * LIB_VERSION[1]: The minor version of the library. 
  * LIB_VERSION[2]: The release number of the library. 


Make sure to update the versions number when a different library is used. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


