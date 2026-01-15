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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title2 "H2")
  * [◆ H5PLappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title3 "H2")
  * [◆ H5PLget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title4 "H2")
  * [◆ H5PLget_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title5 "H2")
  * [◆ H5PLinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title6 "H2")
  * [◆ H5PLprepend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title7 "H2")
  * [◆ H5PLremove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title8 "H2")
  * [◆ H5PLreplace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title9 "H2")
  * [◆ H5PLset_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title10 "H2")
  * [◆ H5PLsize()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#func-members)
Dynamically-loaded Plugins (H5PL)
## Detailed Description
Use the functions in this module to manage the loading behavior of HDF5 plugins. 

Attention
    The loading behavior of HDF5 plugins can be controlled via the functions described below and certain environment variables, such as `HDF5_PLUGIN_PRELOAD` and `HDF5_PLUGIN_PATH`. 
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gac74200fdc02f794f3fae753fe8b850b0) (const char *search_path)  
| Inserts a plugin path at the end of the plugin search path list.   
  
ssize_t  |  [H5PLget](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga64a3c5450d91455624ecba582553d905) (unsigned int index, char *path_buf, size_t buf_size)  
| Queries the plugin search path list at the specified index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLget_loading_state](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gadde42938cf1cc41fa392e8719050b52a) (unsigned int *plugin_control_mask)  
| Queries the loadability of dynamic plugin types.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gacc5153b0db6b3f876c3980bf34f931fc) (const char *search_path, unsigned int index)  
| Inserts a path at the specified index in the plugin search path list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLprepend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga1f2300ef2de6e430af330de7d194576f) (const char *search_path)  
| Inserts a plugin path at the beginning of the plugin search path list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLremove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gaa566196b7c6970c255feac4cf9f3bf40) (unsigned int index)  
| Removes a plugin path at a specified index from the plugin search path list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLreplace](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gab0f8d4e8d0b81cb55cf8b9de5095dc0b) (const char *search_path, unsigned int index)  
| Replaces the path at the specified index in the plugin search path list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLset_loading_state](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga3ee7276ca1168b175661db3866f8db0e) (unsigned int plugin_control_mask)  
| Controls the loadability of dynamic plugin types.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PLsize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga30b799ad7e9645312ef8975a610b4b18) (unsigned int *num_paths)  
| Retrieves the number of stored plugin paths.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gac74200fdc02f794f3fae753fe8b850b0)H5PLappend()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLappend  | ( | const char * | _search_path_ | ) |   
---|---|---|---|---|---  
Inserts a plugin path at the end of the plugin search path list.  

Parameters
     [in] | search_path | A plugin path   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gac74200fdc02f794f3fae753fe8b850b0 "Inserts a plugin path at the end of the plugin search path list.") inserts a plugin path at the end of the plugin search path list. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga64a3c5450d91455624ecba582553d905)H5PLget()
ssize_t H5PLget  | ( | unsigned int |  _index_ ,   
---|---|---|---  
|  | char * |  _path_buf_ ,   
|  | size_t |  _buf_size_ )  
Queries the plugin search path list at the specified index.  

Parameters
     [in] | index | Index   
---|---|---  
[out] | path_buf | Pathname   
[in] | buf_size | Size of `path_buf` 

Returns
    Returns the length of the path, a non-negative value, if successful; otherwise returns a negative value.  
[H5PLget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga64a3c5450d91455624ecba582553d905 "Queries the plugin search path list at the specified index.") queries the plugin path at a specified index. If `path_buf` is non-NULL then it writes up to `buf_size` bytes into that buffer and always returns the length of the path name.
If `path_buf` is NULL, this function will simply return the number of characters required to store the path name, ignoring `path_buf` and `buf_size`.
If an error occurs then the buffer pointed to by `path_buf` (NULL or non-NULL) is unchanged and the function returns a negative value. If a zero is returned for the name's length, then there is no path name associated with the index. and the `path_buf` buffer will be unchanged. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gadde42938cf1cc41fa392e8719050b52a)H5PLget_loading_state()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLget_loading_state  | ( | unsigned int * | _plugin_control_mask_ | ) |   
---|---|---|---|---|---  
Queries the loadability of dynamic plugin types.  

Parameters
     [out] | plugin_control_mask | List of dynamic plugin types that are enabled or disabled.  
A plugin bit set to 0 (zero) indicates that the dynamic plugin type is disabled.  
A plugin bit set to 1 (one) indicates that the dynamic plugin type is enabled.  
If the value of `plugin_control_mask` is negative, all dynamic plugin types are enabled.  
If the value of `plugin_control_mask` is 0 (zero), all dynamic plugins are disabled.   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLget_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gadde42938cf1cc41fa392e8719050b52a "Queries the loadability of dynamic plugin types.") retrieves the bitmask that controls whether a certain type of plugin (e.g.: filters, VOL drivers) will be loaded by the HDF5 library.
Bit positions allocated to date are specified in [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) as follows: 
typedef enum [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) {
[H5PL_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fae335485e955aa13e2fa68e07db2e3849) = -1, 
[H5PL_TYPE_FILTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa07a9b7f82eadd1f4c70ec6e67d8a3f00) = 0, 
[H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0) = 1, 
[H5PL_TYPE_VFD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa6df516700d4cb82915d6fe276cf70629) = 2, 
[H5PL_TYPE_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa90f9d05bec563c3854960a245a27d36e) = 3 
} [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f); 

Since
    1.8.15 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gacc5153b0db6b3f876c3980bf34f931fc)H5PLinsert()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLinsert  | ( | const char * |  _search_path_ ,   
---|---|---|---  
|  | unsigned int |  _index_ )  
Inserts a path at the specified index in the plugin search path list.  

Parameters
     [in] | search_path | A plugin path   
---|---|---  
[in] | index | Index  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gacc5153b0db6b3f876c3980bf34f931fc "Inserts a path at the specified index in the plugin search path list.") inserts a plugin path at the specified index in the plugin search path list, moving other paths after `index`. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga1f2300ef2de6e430af330de7d194576f)H5PLprepend()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLprepend  | ( | const char * | _search_path_ | ) |   
---|---|---|---|---|---  
Inserts a plugin path at the beginning of the plugin search path list.  

Parameters
     [in] | search_path | A plugin path   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLprepend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga1f2300ef2de6e430af330de7d194576f "Inserts a plugin path at the beginning of the plugin search path list.") inserts a plugin path at the end of the plugin search path list. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gaa566196b7c6970c255feac4cf9f3bf40)H5PLremove()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLremove  | ( | unsigned int | _index_ | ) |   
---|---|---|---|---|---  
Removes a plugin path at a specified index from the plugin search path list.  

Parameters
     [in] | index | Index   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLremove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gaa566196b7c6970c255feac4cf9f3bf40 "Removes a plugin path at a specified index from the plugin search path list.") removes a plugin path at the specified `index` and compacts the plugin search path list. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gab0f8d4e8d0b81cb55cf8b9de5095dc0b)H5PLreplace()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLreplace  | ( | const char * |  _search_path_ ,   
---|---|---|---  
|  | unsigned int |  _index_ )  
Replaces the path at the specified index in the plugin search path list.  

Parameters
     [in] | search_path | A plugin path   
---|---|---  
[in] | index | Index  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLreplace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#gab0f8d4e8d0b81cb55cf8b9de5095dc0b "Replaces the path at the specified index in the plugin search path list.") replaces a plugin path at the specified index in the plugin search path list. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga3ee7276ca1168b175661db3866f8db0e)H5PLset_loading_state()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLset_loading_state  | ( | unsigned int | _plugin_control_mask_ | ) |   
---|---|---|---|---|---  
Controls the loadability of dynamic plugin types.  

Parameters
     [in] | plugin_control_mask | The list of dynamic plugin types to enable or disable.  
A plugin bit set to 0 (zero) prevents use of that dynamic plugin.  
A plugin bit set to 1 (one) enables use of that dynamic plugin.  
Setting `plugin_control_mask` to a negative value enables all dynamic plugin types.  
Setting `plugin_control_mask` to 0 (zero) disables all dynamic plugin  
types.   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLset_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga3ee7276ca1168b175661db3866f8db0e "Controls the loadability of dynamic plugin types.") uses one argument to enable or disable individual plugin types.
The `plugin_control_mask` parameter is an encoded integer in which each bit controls a specific plugin type. Bit positions allocated to date are specified in [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) as follows: 
typedef enum [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) {
[H5PL_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fae335485e955aa13e2fa68e07db2e3849) = -1, 
[H5PL_TYPE_FILTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa07a9b7f82eadd1f4c70ec6e67d8a3f00) = 0, 
[H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0) = 1, 
[H5PL_TYPE_VFD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa6df516700d4cb82915d6fe276cf70629) = 2, 
[H5PL_TYPE_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fa90f9d05bec563c3854960a245a27d36e) = 3 
} [H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f);
A plugin bit set to 0 (zero) prevents the use of the dynamic plugin type corresponding to that bit position. A plugin bit set to 1 (one) allows the use of that dynamic plugin type.
All dynamic plugin types can be enabled by setting `plugin_control_mask` to a negative value. A value of 0 (zero) will disable all dynamic plugin types.
The loading of external dynamic plugins can be controlled during runtime with an environment variable, `HDF5_PLUGIN_PRELOAD`. [H5PLset_loading_state()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga3ee7276ca1168b175661db3866f8db0e "Controls the loadability of dynamic plugin types.") inspects the `HDF5_PLUGIN_PRELOAD` environment variable every time it is called. If the environment variable is set to the special `::` string, all dynamic plugins are disabled. 

Warning
    The environment variable `HDF5_PLUGIN_PRELOAD` controls the loading of dynamic plugin types at runtime. If it is set to disable all plugin types, then it will disable them for _all_ running programs that access the same variable instance. 

Since
    1.8.15 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga30b799ad7e9645312ef8975a610b4b18)H5PLsize()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PLsize  | ( | unsigned int * | _num_paths_ | ) |   
---|---|---|---|---|---  
Retrieves the number of stored plugin paths.  

Parameters
     [out] | num_paths | Current length of the plugin search path list   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5PLsize()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html#ga30b799ad7e9645312ef8975a610b4b18 "Retrieves the number of stored plugin paths.") retrieves the number of paths stored in the plugin search path list. 

Since
    1.10.1 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


