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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title2 "H2")
  * [◆ H5Zfilter_avail()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title3 "H2")
  * [◆ H5Zget_filter_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title4 "H2")
  * [◆ H5Zunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#func-members)
Java Filter (H5Z) Interface
## Detailed Description 

See also
     [Filters (H5Z)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html), C-API      [HDF5 Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html), User Guide 
##  Functions  
---  
static synchronized native int  |  [H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#ga3726d95e7c63ee7e9969b6f0c6a37a99) (int filter) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#gaedba9d2e5855cac2db0c2e121aa5e11c) (int filter) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#ga17cde338cf13be6d291ae6215656ca08) (int filter) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#ga3726d95e7c63ee7e9969b6f0c6a37a99)H5Zfilter_avail()
| static synchronized native int H5Zfilter_avail  | ( | int | _filter_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Zfilter_avail checks if a filter is available. 

Parameters
     filter | IN: filter number.  
---|--- 

Returns
    a non-negative(TRUE/FALSE) value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#gaedba9d2e5855cac2db0c2e121aa5e11c)H5Zget_filter_info()
| static synchronized native int H5Zget_filter_info  | ( | int | _filter_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Zget_filter_info gets information about a pipeline data filter. 

Parameters
     filter | IN: filter number.  
---|--- 

Returns
    the filter information flags 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html#ga17cde338cf13be6d291ae6215656ca08)H5Zunregister()
| static synchronized native int H5Zunregister  | ( | int | _filter_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Zunregister unregisters a filter. 

Parameters
     filter | IN: filter number.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


