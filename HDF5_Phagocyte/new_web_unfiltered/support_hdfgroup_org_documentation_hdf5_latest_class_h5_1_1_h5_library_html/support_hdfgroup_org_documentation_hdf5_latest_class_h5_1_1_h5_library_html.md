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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title0 "H2")
  * [ Static Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title1 "H2")
  * [Member Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title2 "H2")
  * [◆ checkVersion()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title3 "H2")
  * [◆ close()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title4 "H2")
  * [◆ dontAtExit()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title5 "H2")
  * [◆ garbageCollect()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title6 "H2")
  * [◆ getLibVersion()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title7 "H2")
  * [◆ initH5cpp()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title8 "H2")
  * [◆ open()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title9 "H2")
  * [◆ setFreeListLimits()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title10 "H2")
  * [◆ termH5cpp()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Static Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#pub-static-methods)
H5Library Class Reference
`#include <c++/src/H5Library.h>`
## Detailed Description
Class [H5Library](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html "Class H5Library operates the HDF5 library globably.") operates the HDF5 library globably. 
It is not necessary to construct an instance of [H5Library](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html "Class H5Library operates the HDF5 library globably.") to use the methods. 
##  Static Public Member Functions  
---  
static void  |  [checkVersion](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a80c319ca94e8b9d58cbe4967cfebbeb6) (unsigned majnum, unsigned minnum, unsigned relnum)  
| Verifies that the arguments match the version numbers compiled into the library.   
  
static void  |  [close](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a5ae591df94fc66ccb85cbb6565368bca) ()  
| Flushes all data to disk, closes files, and cleans up memory.   
  
static void  |  [dontAtExit](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a65b84e44635b954942c7e9da810b2086) ()  
| Instructs library not to install the C `atexit` cleanup routine.   
  
static void  |  [garbageCollect](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a39d77be394445d2771dafe1cef483436) ()  
| Walks through all the garbage collection routines for the library, which are supposed to free any unused memory they have allocated.   
  
static void  |  [getLibVersion](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a01b5622d859d19759f2ac3d441a2e4fa) (unsigned &majnum, unsigned &minnum, unsigned &relnum)  
| Returns the HDF library release number.   
  
static void  |  [initH5cpp](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#ae347d2b5c83cdfb9705d7dc5b838c42a) (void)  
| Initializes C++ library and registers terminating functions at exit. Only for the library functions, not for user-defined functions.   
  
static void  |  [open](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a9e8555112049fc2b4945120b3c45f8ab) ()  
| Initializes the HDF5 library.   
  
static void  |  [setFreeListLimits](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a5c3addd1bf267107206280a31cd51fd7) (int reg_global_lim, int reg_list_lim, int arr_global_lim, int arr_list_lim, int blk_global_lim, int blk_list_lim)  
| Sets limits on the different kinds of free lists.   
  
static void  |  [termH5cpp](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#aaac8986dfafb237854a5856c73e174ab) (void)  
| Sends request for the C layer to terminate.   
  
## Member Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a80c319ca94e8b9d58cbe4967cfebbeb6)checkVersion()
| void checkVersion  | ( | unsigned |  _majnum_ ,   
---|---|---|---  
|  | unsigned |  _minnum_ ,   
|  | unsigned |  _relnum_ )  
static  
Verifies that the arguments match the version numbers compiled into the library.  

Parameters
     majnum | - IN: Major version of the library   
---|---  
minnum | - IN: Minor version of the library   
relnum | - IN: Release number of the library  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|--- 

Description
    For information about library version, please refer to the H5check_version API in the HDF5 C Reference Manual.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a5ae591df94fc66ccb85cbb6565368bca)close()
| void close  | ( |  | ) |   
---|---|---|---|---  
static  
Flushes all data to disk, closes files, and cleans up memory.  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a65b84e44635b954942c7e9da810b2086)dontAtExit()
| void dontAtExit  | ( |  | ) |   
---|---|---|---|---  
static  
Instructs library not to install the C `atexit` cleanup routine.  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a39d77be394445d2771dafe1cef483436)garbageCollect()
| void garbageCollect  | ( |  | ) |   
---|---|---|---|---  
static  
Walks through all the garbage collection routines for the library, which are supposed to free any unused memory they have allocated.  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|--- 

Description
    It is not required that [H5Library::garbageCollect](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a39d77be394445d2771dafe1cef483436 "Walks through all the garbage collection routines for the library, which are supposed to free any unu...") be called at any particular time; it is only necessary in certain situations, such as when the application has performed actions that cause the library to allocate many objects. The application should call [H5Library::garbageCollect](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a39d77be394445d2771dafe1cef483436 "Walks through all the garbage collection routines for the library, which are supposed to free any unu...") if it eventually releases those objects and wants to reduce the memory used by the library from the peak usage required.      The library automatically garbage collects all the free lists when the application ends.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a01b5622d859d19759f2ac3d441a2e4fa)getLibVersion()
| void getLibVersion  | ( | unsigned & |  _majnum_ ,   
---|---|---|---  
|  | unsigned & |  _minnum_ ,   
|  | unsigned & |  _relnum_ )  
static  
Returns the HDF library release number.  

Parameters
     majnum | - OUT: Major version of the library   
---|---  
minnum | - OUT: Minor version of the library   
relnum | - OUT: Release number of the library  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#ae347d2b5c83cdfb9705d7dc5b838c42a)initH5cpp()
| void initH5cpp  | ( | void |  | ) |   
---|---|---|---|---|---  
static  
Initializes C++ library and registers terminating functions at exit. Only for the library functions, not for user-defined functions.  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a9e8555112049fc2b4945120b3c45f8ab)open()
| void open  | ( |  | ) |   
---|---|---|---|---  
static  
Initializes the HDF5 library.  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#a5c3addd1bf267107206280a31cd51fd7)setFreeListLimits()
| void setFreeListLimits  | ( | int |  _reg_global_lim_ ,   
---|---|---|---  
|  | int |  _reg_list_lim_ ,   
|  | int |  _arr_global_lim_ ,   
|  | int |  _arr_list_lim_ ,   
|  | int |  _blk_global_lim_ ,   
|  | int |  _blk_list_lim_ )  
static  
Sets limits on the different kinds of free lists.  

Parameters
     reg_global_lim | - IN: Limit on all "regular" free list memory used   
---|---  
reg_list_lim | - IN: Limit on memory used in each "regular" free list   
arr_global_lim | - IN: Limit on all "array" free list memory used   
arr_list_lim | - IN: Limit on memory used in each "array" free list   
blk_global_lim | - IN: Limit on all "block" free list memory used   
blk_list_lim | - IN: Limit on memory used in each "block" free list  

Exceptions
     [H5::LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html) |   
---|--- 

Description
    Setting a value of -1 for a limit means no limit of that type. For more information on free list limits, please refer to the H5set_free_list_limits API in the HDF5 C Reference Manual.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#aaac8986dfafb237854a5856c73e174ab)termH5cpp()
| void termH5cpp  | ( | void |  | ) |   
---|---|---|---|---|---  
static  
Sends request for the C layer to terminate.  

Description
    If the C library fails to terminate, exit with a failure. 
* * *
The documentation for this class was generated from the following files:
  * c++/src/[H5Library.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_library_8h.html)
  * c++/src/[H5Library.cpp](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_library_8cpp.html)


  * [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)
  * [H5Library](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


