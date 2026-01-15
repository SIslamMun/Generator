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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title2 "H2")
  * [◆ H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title3 "H2")
  * [◆ H5atclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title4 "H2")
  * [◆ H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title5 "H2")
  * [◆ H5close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title6 "H2")
  * [◆ H5dont_atexit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title7 "H2")
  * [◆ H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title8 "H2")
  * [◆ H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title9 "H2")
  * [◆ H5get_free_list_sizes()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title10 "H2")
  * [◆ H5get_libversion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title11 "H2")
  * [◆ H5is_library_terminating()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title12 "H2")
  * [◆ H5is_library_threadsafe()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title13 "H2")
  * [◆ H5open()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title14 "H2")
  * [◆ H5resize_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title15 "H2")
  * [◆ H5set_free_list_limits()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#title16 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#func-members)
Library General (H5)
## Detailed Description
Use the functions in this module to manage the life cycle of HDF5 library instances.
Create | Read   
---|---  
23 { 24 // an HDF5 library instance is automatically initialized as 25 // part of the first HDF5 API call, but there's no harm in 26 // calling H5open(). 27 if ([H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480)() < 0) { 28 ret_val = EXIT_FAILURE; 29 } 30 } |  34 { 35 __label__ fail_read; 36 unsigned majnum, minnum, relnum; 37 bool flag; 38 39 // retrieve the library version 40 if ([H5get_libversion](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaf87da966fdf896ec7bca794e21d4ab0a)(&majnum, &minnum, &relnum) < 0) { 41 ret_val = EXIT_FAILURE; 42 goto fail_read; 43 } 44 // is this a thread-safe library build? 45 if ([H5is_library_threadsafe](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaae7a734fe6c8080bff45b9142a173bc5)(&flag) < 0) { 46 ret_val = EXIT_FAILURE; 47 goto fail_read; 48 } 49 50 printf("Welcome to HDF5 %d.%d.%d\n", majnum, minnum, relnum); 51 printf("Thread-safety %s\n", (flag > 0) ? "enabled" : "disabled"); 52 53fail_read:; 54 }  
Update | Delete   
58 { 59 // update the library instance free list limits 60 if ([H5set_free_list_limits](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545)(512 * 1024, 32 * 1024, 2048 * 1024, 128 * 1024, 8192 * 1024, 512 * 1024) < 61 0) { 62 ret_val = EXIT_FAILURE; 63 } 64 } |  10void 11closing_shop(void *ctx) 12{ 13 printf("GoodBye, Cruel World!\n"); 14} 68 { 69#if H5_VERSION_GE(1, 13, 0) 70 // install a finalization routine 71 if ([H5atclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gade62a3e6b75bed52234d9e8432c27fb2)(&closing_shop, NULL) < 0) { 72 ret_val = EXIT_FAILURE; 73 } 74#endif 75 // close shop 76 if ([H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574)() < 0) { 77 ret_val = EXIT_FAILURE; 78 } 79 }  
##  Functions  
---  
void *  |  [H5allocate_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683) (size_t size, bool clear)  
| Allocates memory that will be freed later internally.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5atclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gade62a3e6b75bed52234d9e8432c27fb2) ([H5_atclose_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a05a43aeba55b736316e74e3a26cbb254) func, void *ctx)  
| Registers a callback for the library to invoke when it's closing.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5check_version](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be) (unsigned majnum, unsigned minnum, unsigned relnum)  
| Verifies that HDF5 library versions are consistent.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574) (void)  
| Flushes all data to disk, closes all open objects, and releases memory.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5dont_atexit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga7f80eb63b5e78812b9d0d50ac46764e8) (void)  
| Instructs library not to install atexit() cleanup routine.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf) (void *mem)  
| Frees memory allocated by the HDF5 library.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5garbage_collect](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef) (void)  
| Garbage collects on all free-lists of all types.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5get_free_list_sizes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga2310d963a6f48ec12fda8c0c8bbefbbb) (size_t *reg_size, size_t *arr_size, size_t *blk_size, size_t *fac_size)  
| Gets the current size of the free lists used to manage memory.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5get_libversion](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaf87da966fdf896ec7bca794e21d4ab0a) (unsigned *majnum, unsigned *minnum, unsigned *relnum)  
| Returns the HDF library release number.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5is_library_terminating](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaff4f67ea7e7e366fcced7dfc3cc87b16) (bool *is_terminating)  
| Checks whether the HDF5 library is closing.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5is_library_threadsafe](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaae7a734fe6c8080bff45b9142a173bc5) (bool *is_ts)  
| Determines whether the HDF5 library was built with the thread-safety feature enabled.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480) (void)  
| Initializes the HDF5 library.   
  
void *  |  [H5resize_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gab8024266aca102414ba2960e27ddb7c2) (void *mem, size_t size)  
| Resizes and, if required, re-allocates memory that will later be freed internally by the HDF5 library.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5set_free_list_limits](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545) (int reg_global_lim, int reg_list_lim, int arr_global_lim, int arr_list_lim, int blk_global_lim, int blk_list_lim)  
| Sets free-list size limits.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683)H5allocate_memory()
void * H5allocate_memory  | ( | size_t |  _size_ ,   
---|---|---|---  
|  | bool |  _clear_ )  
Allocates memory that will be freed later internally.  

Parameters
     [in] | size | The size in bytes of the buffer to be allocated   
---|---|---  
[in] | clear | Flag whether the new buffer is to be initialized with 0 

Returns
    On success, returns pointer to newly allocated buffer or returns NULL if size is 0 (zero).  
Returns NULL on failure.  
[H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally.") allocates a memory buffer of size bytes that will later be freed internally by the HDF5 library.
The boolean `clear` parameter specifies whether the buffer should be initialized. If clear is `true`, all bits in the buffer are to be set to 0 (zero); if clear is `false`, the buffer will not be initialized.
This function is intended to have the semantics of malloc() and calloc(). However, unlike malloc() and calloc(), which allow for a "special" pointer to be returned instead of NULL, this function always returns NULL on failure or when size is set to 0 (zero). 

Note
    At this time, the only intended use for this function is to allocate memory that will be returned to the library as a data buffer from a third-party filter. 

Attention
    To avoid heap corruption, allocated memory should be freed using the same library that initially allocated it. In most cases, the HDF5 API uses resources that are allocated and freed either entirely by the user or entirely by the library, so this is not a problem. In rare cases, however, HDF5 API calls will free the memory that the user allocated. This function allows the user to safely allocate this memory.  
It is particularly important to use this function to allocate memory in Microsoft Windows environments. In Windows, the C standard library is implemented in dynamic link libraries (DLLs) known as the C run-time (CRT). Each version of Visual Studio comes with multiple versions of the CRT DLLs (debug, release, et cetera) and allocating and freeing memory across DLL boundaries can cause resource leaks and subtle bugs due to heap corruption.  
Even when using this function, it is best where possible to ensure that all components of a C application are built with the same version of Visual Studio and configuration (Debug or Release), and thus linked against the same CRT.  
Use this function only to allocate memory inside third-party HDF5 filters. It will generally not be safe to use this function to allocate memory for any other purpose. 

See also
     [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library."), [H5resize_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gab8024266aca102414ba2960e27ddb7c2 "Resizes and, if required, re-allocates memory that will later be freed internally by the HDF5 library...") 

Since
    1.8.15 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gade62a3e6b75bed52234d9e8432c27fb2)H5atclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5atclose  | ( | [H5_atclose_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a05a43aeba55b736316e74e3a26cbb254) |  _func_ ,   
---|---|---|---  
|  | void * |  _ctx_ )  
Registers a callback for the library to invoke when it's closing.  

Parameters
     [in] | func | The function pointer to invoke   
---|---|---  
[in] | ctx | Context to pass to `func` when invoked  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5atclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gade62a3e6b75bed52234d9e8432c27fb2 "Registers a callback for the library to invoke when it's closing.") registers a callback that the HDF5 library will invoke when closing. The full capabilities of the HDF5 library are available to callbacks invoked through this mechanism, and library shutdown will only begin in earnest when all callbacks have been invoked and have returned.
Registered callbacks are invoked in LIFO order, similar to the Standard C 'atexit' routine. For example, if 'func1' is registered, then 'func2', when the library is closing 'func2', will be invoked first, then 'func1'.
The `ctx` pointer will be passed to `func` when it's invoked. NULL is allowed for `ctx`.
If the HDF5 library is initialized and closed more than once, the `func` callback must be registered within each open/close cycle. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be)H5check_version()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5check_version  | ( | unsigned |  _majnum_ ,   
---|---|---|---  
|  | unsigned |  _minnum_ ,   
|  | unsigned |  _relnum_ )  
Verifies that HDF5 library versions are consistent.  

Parameters
     [in] | majnum | HDF5 library major version number   
---|---|---  
[in] | minnum | HDF5 library minor version number   
[in] | relnum | HDF5 library release number  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.") verifies that the version of the HDF5 library with which an application was compiled, as indicated by the passed parameters, matches the version of the HDF5 library against which the application is currently linked.
`majnum` is the major version number of the HDF library with which the application was compiled, `minnum` is the minor version number, and `relnum` is the release number. Consider the following example:
An official HDF5 release is labelled as follows: HDF5 Release `<majnum>.<minnum>.<relnum>`  
For example, in HDF5 Release 1.8.5: 
  * 1 is the major version number, `majnum`. 
  * 8 is the minor version number, `minnum`. 
  * 5 is the release number, `relnum`.


As stated above, [H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.") first verifies that the version of the HDF5 library with which an application was compiled matches the version of the HDF5 library against which the application is currently linked. If this check fails, [H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.") causes the application to abort (by means of a standard C abort() call) and prints information that is usually useful for debugging. This precaution is taken to avoid the risks of data corruption or segmentation faults.
The most common cause of this failure is that an application was compiled with one version of HDF5 and is dynamically linked with a different version different version.
If the above test passes, [H5check_version()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga93525482e1168344f4c92470f99d88be "Verifies that HDF5 library versions are consistent.") proceeds to verify the consistency of additional library version information. This is designed to catch source code inconsistencies that do not normally cause failures; if this check reveals an inconsistency, an informational warning is printed but the application is allowed to run. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574)H5close()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5close  | ( | void |  | ) |   
---|---|---|---|---|---  
Flushes all data to disk, closes all open objects, and releases memory.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.
[H5close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") flushes all data to disk, closes all open HDF5 objects, and cleans up all memory used by the HDF5 library. This function is generally called when the application calls exit(), but may be called earlier in the event of an emergency shutdown or out of a desire to free all resources used by the HDF5 library. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga7f80eb63b5e78812b9d0d50ac46764e8)H5dont_atexit()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5dont_atexit  | ( | void |  | ) |   
---|---|---|---|---|---  
Instructs library not to install atexit() cleanup routine.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.
[H5dont_atexit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga7f80eb63b5e78812b9d0d50ac46764e8 "Instructs library not to install atexit\(\) cleanup routine.") indicates to the library that an atexit() cleanup routine should not be installed. The major purpose for using this function is in situations where the library is dynamically linked into an application and is un-linked from the application before exit() gets called. In those situations, a routine installed with atexit() would jump to a routine that was no longer in memory, causing errors. 

Attention
    In order to be effective, this routine _must_ be called before any other HDF5 function calls, and must be called each time the library is loaded/linked into the application (the first time and after it's been unloaded). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)H5free_memory()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5free_memory  | ( | void * | _mem_ | ) |   
---|---|---|---|---|---  
Frees memory allocated by the HDF5 library.  

Parameters
     [in] | mem | Buffer to be freed. Can be NULL   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") frees the memory that has been allocated by the caller with [H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally.") or by the HDF5 library on behalf of the caller.
[H5Tget_member_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66 "Retrieves the name of a compound or enumeration datatype member.") provides an example of memory allocation on behalf of the caller: The function returns a buffer containing the name of a compound datatype member. It is the caller's responsibility to eventually free that buffer with [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library."). 

Attention
    It is especially important to use this function to free memory allocated by the library on Windows. The C standard library is implemented in dynamic link libraries (DLLs) known as the C run-time (CRT). Each version of Visual Studio comes with two CRT DLLs (debug and release) and allocating and freeing across DLL boundaries can cause resource leaks and subtle bugs due to heap corruption.  
Only use this function to free memory allocated by the HDF5 Library. It will generally not be safe to use this function to free memory allocated by any other means.  
Even when using this function, it is still best to ensure that all components of a C application are built with the same version of Visual Studio and build (debug or release) and thus linked against the same CRT. 

See also
     [H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally."), [H5resize_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gab8024266aca102414ba2960e27ddb7c2 "Resizes and, if required, re-allocates memory that will later be freed internally by the HDF5 library...") 

Since
    1.8.13 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef)H5garbage_collect()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5garbage_collect  | ( | void |  | ) |   
---|---|---|---|---|---  
Garbage collects on all free-lists of all types.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.
[H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef "Garbage collects on all free-lists of all types.") walks through all garbage collection routines of the library, freeing any unused memory.
It is not required that [H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef "Garbage collects on all free-lists of all types.") be called at any particular time; it is only necessary for certain situations where the application has performed actions that cause the library to allocate many objects. The application should call [H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef "Garbage collects on all free-lists of all types.") if it eventually releases those objects and wants to reduce the memory used by the library from the peak usage required. 

Note
    The library automatically garbage collects all the free lists when the application ends. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga2310d963a6f48ec12fda8c0c8bbefbbb)H5get_free_list_sizes()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5get_free_list_sizes  | ( | size_t * |  _reg_size_ ,   
---|---|---|---  
|  | size_t * |  _arr_size_ ,   
|  | size_t * |  _blk_size_ ,   
|  | size_t * |  _fac_size_ )  
Gets the current size of the free lists used to manage memory.  

Parameters
     [out] | reg_size | The current size of all "regular" free list memory used   
---|---|---  
[out] | arr_size | The current size of all "array" free list memory used   
[out] | blk_size | The current size of all "block" free list memory used   
[out] | fac_size | The current size of all "factory" free list memory used  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5get_free_list_sizes()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga2310d963a6f48ec12fda8c0c8bbefbbb "Gets the current size of the free lists used to manage memory.") obtains the current size of the different kinds of free lists that the library uses to manage memory. The free list sizes can be set with [H5set_free_list_limits()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545 "Sets free-list size limits.") and garbage collected with [H5garbage_collect()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gae511943bcb837a52a012a3a5dd7b90ef "Garbage collects on all free-lists of all types."). These lists are global for the entire library. 

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaf87da966fdf896ec7bca794e21d4ab0a)H5get_libversion()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5get_libversion  | ( | unsigned * |  _majnum_ ,   
---|---|---|---  
|  | unsigned * |  _minnum_ ,   
|  | unsigned * |  _relnum_ )  
Returns the HDF library release number.  

Parameters
     [out] | majnum | The major version number of the library   
---|---|---  
[out] | minnum | The minor version number of the library   
[out] | relnum | The release version number of the library  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5get_libversion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaf87da966fdf896ec7bca794e21d4ab0a "Returns the HDF library release number.") retrieves the major, minor, and release numbers of the version of the HDF5 library which is linked to the application. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaff4f67ea7e7e366fcced7dfc3cc87b16)H5is_library_terminating()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5is_library_terminating  | ( | bool * | _is_terminating_ | ) |   
---|---|---|---|---|---  
Checks whether the HDF5 library is closing.  

Parameters
     [out] | is_terminating | Flag indicating whether library is shutting down   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5is_library_terminating()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaff4f67ea7e7e366fcced7dfc3cc87b16 "Checks whether the HDF5 library is closing.") queries whether the HDF5 library is in the process of shutting down. The `is_terminating` flag will only be set to true after shutdown starts, it will be false before the library has been initialized, while the library is initialized, and after it has been closed. The value of `is_terminating` is undefined if this routine fails. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaae7a734fe6c8080bff45b9142a173bc5)H5is_library_threadsafe()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5is_library_threadsafe  | ( | bool * | _is_ts_ | ) |   
---|---|---|---|---|---  
Determines whether the HDF5 library was built with the thread-safety feature enabled.  

Parameters
     [out] | is_ts | Boolean value indicating whether the library was built with thread-safety enabled   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The HDF5 library, although not internally multi-threaded, can be built with a thread-safety feature enabled that protects internal data structures with a mutex. In certain circumstances, it may be useful to determine, at run-time, whether the linked HDF5 library was built with the thread-safety feature enabled. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480)H5open()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5open  | ( | void |  | ) |   
---|---|---|---|---|---  
Initializes the HDF5 library.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.
[H5open()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library.") initializes the HDF5 library.
When the HDF5 library is used in a C application, the library is automatically initialized when the first HDF5 function call is issued. If one finds that an HDF5 library function is failing inexplicably, [H5open()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library.") can be called first. It is safe to call [H5open()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library.") before an application issues any other function calls to the HDF5 library, as there are no damaging side effects in calling it more than once. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gab8024266aca102414ba2960e27ddb7c2)H5resize_memory()
void * H5resize_memory  | ( | void * |  _mem_ ,   
---|---|---|---  
|  | size_t |  _size_ )  
Resizes and, if required, re-allocates memory that will later be freed internally by the HDF5 library.  

Parameters
     [in] | mem | Pointer to a buffer to be resized. May be NULL   
---|---|---  
[in] | size | New size of the buffer, in bytes 

Returns
    On success, returns pointer to resized or reallocated buffer or returns NULL if size is 0 (zero).  
Returns NULL on failure.  
[H5resize_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gab8024266aca102414ba2960e27ddb7c2 "Resizes and, if required, re-allocates memory that will later be freed internally by the HDF5 library...") takes a pointer to an existing buffer and resizes the buffer to match the value in `size`. If necessary, the buffer is reallocated. If `size` is 0, the buffer is released.
The input buffer must either be NULL or have been allocated by [H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally.") since the input buffer may be freed by the library.
For certain behaviors, the pointer `mem` may be passed in as NULL.
This function is intended to have the semantics of realloc():
`H5resize_memory(buffer, size)` | Resizes buffer. Returns pointer to resized buffer.   
---|---  
`H5resize_memory(NULL, size)` | Allocates memory using HDF5 Library allocator. Returns pointer to new buffer   
`H5resize_memory(buffer, 0)` | Frees memory using HDF5 Library allocator. Returns NULL.   
`H5resize_memory(NULL, 0)` | Returns NULL (undefined in C standard).   
Unlike realloc(), which allows for a "special pointer to be returned instead of NULL, this function always returns NULL on failure or when size is 0 (zero). 

Note
    At this time, the only intended use for this function is to resize or reallocate memory that will be returned to the library (and eventually to the user) as a data buffer from a third-party HDF5 filter. 

Attention
    To avoid heap corruption, allocated memory should be freed using the same library that initially allocated it. In most cases, the HDF5 API uses resources that are allocated and freed either entirely by the user or entirely by the library, so this is not a problem. In rare cases, however, HDF5 API calls will free memory that the user allocated. This function allows the user to safely allocate this memory.  
It is particularly important to use this function to resize memory on Microsoft Windows systems. In Windows, the C standard library is implemented in dynamic link libraries (DLLs) known as the C run-time (CRT). Each version of Visual Studio comes with multiple versions of the CRT DLLs (debug, release, et cetera) and allocating and freeing memory across DLL boundaries can cause resource leaks and subtle bugs due to heap corruption.  
Even when using this function, it is still best to ensure that all components of a C application are built with the same version of Visual Studio and the same configuration (Debug or Release), and thus linked against the same CRT.  
Only use this function to resize memory inside third-party HDF5 filters. It will generally not be safe to use this function to resize memory for any other purpose. 

See also
     [H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally."), [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") 

Since
    1.8.15 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545)H5set_free_list_limits()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5set_free_list_limits  | ( | int |  _reg_global_lim_ ,   
---|---|---|---  
|  | int |  _reg_list_lim_ ,   
|  | int |  _arr_global_lim_ ,   
|  | int |  _arr_list_lim_ ,   
|  | int |  _blk_global_lim_ ,   
|  | int |  _blk_list_lim_ )  
Sets free-list size limits.  

Parameters
     [in] | reg_global_lim | The cumulative limit, in bytes, on memory used for all regular free lists (Default: 1MB)   
---|---|---  
[in] | reg_list_lim | The limit, in bytes, on memory used for each regular free list (Default: 64KB)   
[in] | arr_global_lim | The cumulative limit, in bytes, on memory used for all array free lists (Default: 4MB)   
[in] | arr_list_lim | The limit, in bytes, on memory used for each array free list (Default: 256KB)   
[in] | blk_global_lim | The cumulative limit, in bytes, on memory used for all block free lists and, separately, for all factory free lists (Default: 16MB)   
[in] | blk_list_lim | The limit, in bytes, on memory used for each block or factory free list (Default: 1MB)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5set_free_list_limits()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#gaa3f78b24b8a1ff4168db2b7ddca21545 "Sets free-list size limits.") sets size limits on all types of free lists. The HDF5 library uses free lists internally to manage memory. The types of free lists used are as follows: 
  * Regular free lists manage memory for single internal data structures. 
  * Array free lists manage memory for arrays of internal data structures. 
  * Block free lists manage memory for arbitrarily-sized blocks of bytes. 
  * Factory free lists manage memory for fixed-size blocks of bytes.


The parameters specify global and per-list limits; for example, `reg_global_limit` and `reg_list_limit` limit the accumulated size of all regular free lists and the size of each individual regular free list, respectively. Therefore, if an application sets a 1Mb limit on each of the global lists, up to 4Mb of total storage might be allocated, 1Mb for each of the regular, array, block, and factory type lists.
The settings specified for block free lists are duplicated for factory free lists. Therefore, increasing the global limit on block free lists by x bytes will increase the potential free list memory usage by 2x bytes.
Using a value of -1 for a limit means that no limit is set for the specified type of free list. 

Version
    1.8.3 Function changed in this release to set factory free list memory limits. 

Since
    1.4.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


