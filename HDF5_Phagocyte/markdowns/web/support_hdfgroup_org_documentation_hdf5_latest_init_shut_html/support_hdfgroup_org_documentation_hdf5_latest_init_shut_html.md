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
  * [ Application perspective](https://support.hdfgroup.org/documentation/hdf5/latest/_init_shut.html#title0 "H1")
  * [ Implicit initialization and shutdown](https://support.hdfgroup.org/documentation/hdf5/latest/_init_shut.html#title1 "H2")
  * [ Explicit initialization and shutdown](https://support.hdfgroup.org/documentation/hdf5/latest/_init_shut.html#title2 "H2")
  * [ Library internals perspective](https://support.hdfgroup.org/documentation/hdf5/latest/_init_shut.html#title3 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Library initialization and shutdown
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Application perspective
##  Implicit initialization and shutdown
When a developer exports a new symbol as part of the HDF5 library, they should make sure that an application cannot enter the library in an uninitialized state through a new API function, or read an uninitialized value from a non-function HDF5 symbol.
The HDF5 library initializes itself when an application either enters the library through an API function call such as [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."), or when an application evaluates an HDF5 symbol that represents either a property-list identifier such as [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) or [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), a property-list class identifier such as [H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e), a VFD identifier such as [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6) or [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606), or a type identifier such as [H5T_NATIVE_INT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga8fa995f599c6c8203a8d4e7a7dd01805).
The library sets a flag when initialization occurs and as long as the flag is set, skips initialization.
The library provides a couple of macros that initialize the library as necessary. The library is initialized as a side-effect of the `FUNC_ENTER_API*` macros used at the top of most API functions. HDF5 library symbols other than functions are provided through `#define`s that use [H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) to introduce a library-initialization call ([H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library.")) at each site where a non-function symbol is used.
Ordinarily the library registers an `atexit(3)` handler to shut itself down when the application exits.
##  Explicit initialization and shutdown
An application may use an API call, [H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480 "Initializes the HDF5 library."), to explicitly initialize the library. [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") explicitly shuts down the library.
#  Library internals perspective
No matter how library initialization begins, eventually the internal function `H5_init_library` will be called. `H5_init_library` is responsible for calling the initializers for every internal HDF5 library module (aka "package") in the correct order so that no module is initialized before its prerequisite modules. A table in `H5_init_library` establishes the order of initialization. If a developer adds a module to the library that it is appropriate to initialize with the rest of the library, then they should insert its initializer into the right place in the table.
`H5_term_library` drives library shutdown. Library shutdown is table-driven, too. If a developer adds a module that needs to release resources during library shutdown, then they should add a call at the right place to the shutdown table. Note that some entries in the shutdown table are marked as "barriers," and if a new module should only be shutdown **strictly after** the preceding modules, then it should be marked as a barrier. See the comments in `H5_term_library` for more information.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


