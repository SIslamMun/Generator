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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title2 "H2")
  * [◆ H5PLappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title3 "H2")
  * [◆ H5PLget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title4 "H2")
  * [◆ H5PLget_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title5 "H2")
  * [◆ H5PLinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title6 "H2")
  * [◆ H5PLprepend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title7 "H2")
  * [◆ H5PLremove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title8 "H2")
  * [◆ H5PLreplace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title9 "H2")
  * [◆ H5PLset_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title10 "H2")
  * [◆ H5PLsize()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#func-members)
Java Plugin (H5PL) Interface
## Detailed Description 

See also
     [Dynamically-loaded Plugins (H5PL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html), C-API      [HDF5 Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html), User Guide 
##  Functions  
---  
static synchronized native void  |  [H5PLappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga1c81591d0f7f74f6addf22fe1b26a724) (String plugin_path) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native String  |  [H5PLget](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga8ba0158e2050ad5a4c673dee97648670) (int index) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5PLget_loading_state](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gabc5575d30abd0e12fc384d2a72e0d77d) () throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5PLinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gaf05018428cac6b3de1f769687aefbd6c) (String plugin_path, int index) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5PLprepend](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga877ff026cf5a3be493c71e06b543c1f6) (String plugin_path) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5PLremove](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga5cb98e6c7b99fe31e1d84d1404de5a53) (int index) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5PLreplace](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga1cd24b01797d556dacf66d04dc068499) (String plugin_path, int index) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5PLset_loading_state](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gac915e1a0dadcb33ae907fce5138a67b4) (int plugin_flags) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5PLsize](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga142d7d39d2ed2f5ff06997701980d29c) () throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga1c81591d0f7f74f6addf22fe1b26a724)H5PLappend()
| static synchronized native void H5PLappend  | ( | String | _plugin_path_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5PLappend inserts the plugin path at the end of the table. 

Parameters
     plugin_path | IN: Path for location of filter plugin libraries.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga8ba0158e2050ad5a4c673dee97648670)H5PLget()
| static synchronized native String H5PLget  | ( | int | _index_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5PLget retrieves the plugin path at the specified index. 

Parameters
     index | IN: The table index (0-based).  
---|--- 

Returns
    the current path at the index in plugin path table 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gabc5575d30abd0e12fc384d2a72e0d77d)H5PLget_loading_state()
| static synchronized native int H5PLget_loading_state  | ( |  | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---  
static  
H5PLget_loading_state retrieves the state of the dynamic plugins flag, plugin_flags.. 

Returns
    the list of dynamic plugin types that are enabled or disabled. A plugin bit set to 0 (zero) indicates that that dynamic plugin is disabled. A plugin bit set to 1 (one) indicates that that dynamic plugin is enabled. If the value of plugin_flags is negative, all dynamic plugins are enabled. If the value of plugin_flags is 0 (zero), all dynamic plugins are disabled. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gaf05018428cac6b3de1f769687aefbd6c)H5PLinsert()
| static synchronized native void H5PLinsert  | ( | String |  _plugin_path_ ,   
---|---|---|---  
|  | int |  _index_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5PLinsert inserts the plugin path at the specified index. 

Parameters
     plugin_path | IN: Path for location of filter plugin libraries.   
---|---  
index | IN: The table index (0-based). 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga877ff026cf5a3be493c71e06b543c1f6)H5PLprepend()
| static synchronized native void H5PLprepend  | ( | String | _plugin_path_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5PLprepend inserts the plugin path at the beginning of the table. 

Parameters
     plugin_path | IN: Path for location of filter plugin libraries.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga5cb98e6c7b99fe31e1d84d1404de5a53)H5PLremove()
| static synchronized native void H5PLremove  | ( | int | _index_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5PLremove removes the plugin path at the specified index. 

Parameters
     index | IN: The table index (0-based).  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga1cd24b01797d556dacf66d04dc068499)H5PLreplace()
| static synchronized native void H5PLreplace  | ( | String |  _plugin_path_ ,   
---|---|---|---  
|  | int |  _index_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5PLreplace replaces the plugin path at the specified index. 

Parameters
     plugin_path | IN: Path for location of filter plugin libraries.   
---|---  
index | IN: The table index (0-based). 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#gac915e1a0dadcb33ae907fce5138a67b4)H5PLset_loading_state()
| static synchronized native void H5PLset_loading_state  | ( | int | _plugin_flags_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5PLset_loading_state uses one argument to enable or disable individual plugins. The plugin_flags parameter is an encoded integer in which each bit controls a specific plugin or class of plugins. A plugin bit set to 0 (zero) prevents the use of the dynamic plugin corresponding to that bit position. A plugin bit set to 1 (one) allows the use of that dynamic plugin. All dynamic plugins can be enabled by setting plugin_flags to a negative value. A value of 0 (zero) will disable all dynamic plugins.
H5PLset_loading_state inspects the HDF5_PLUGIN_PRELOAD environment variable every time it is called. If the environment variable is set to the special :: string, all dynamic plugins will be disabled. 

Parameters
     plugin_flags | IN: The list of dynamic plugin types to enable or disable. A plugin bit set to 0 (zero) prevents use of that dynamic plugin. A plugin bit set to 1 (one) enables use of that dynamic plugin. Setting plugin_flags to a negative value enables all dynamic plugins. Setting plugin_flags to 0 (zero) disables all dynamic plugins.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html#ga142d7d39d2ed2f5ff06997701980d29c)H5PLsize()
| static synchronized native int H5PLsize  | ( |  | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---  
static  
H5PLsize retrieves the size of the current list of plugin paths. 

Returns
    the current number of paths in the plugin path table 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


