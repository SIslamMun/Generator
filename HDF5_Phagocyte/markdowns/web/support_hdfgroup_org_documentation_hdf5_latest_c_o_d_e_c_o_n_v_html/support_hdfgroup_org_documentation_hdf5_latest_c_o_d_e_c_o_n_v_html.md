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
  * [ Function / Variable Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_c_o_d_e_c_o_n_v.html#title0 "H1")
  * [ Functions that accept printf(3) and scanf(3) format strings](https://support.hdfgroup.org/documentation/hdf5/latest/_c_o_d_e_c_o_n_v.html#title1 "H2")
  * [ Functions that do never return](https://support.hdfgroup.org/documentation/hdf5/latest/_c_o_d_e_c_o_n_v.html#title2 "H2")
  * [ Other attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_c_o_d_e_c_o_n_v.html#title3 "H2")
  * [ Unused variables and parameters](https://support.hdfgroup.org/documentation/hdf5/latest/_c_o_d_e_c_o_n_v.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Library Code Conventions
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
This document describes some practices that are new, or newly documented, starting in 2020.
#  Function / Variable Attributes
In H5private.h, the library provides platform-independent macros for qualifying function and variable definitions.
##  Functions that accept `printf(3)` and `scanf(3)` format strings
Label functions that accept a `printf(3)`-compliant format string with `H5_ATTR_FORMAT(printf,format_argno,variadic_argno)`, where the format string is the `format_argno`th argument (counting from 1) and the variadic arguments start with the `variadic_argno`th.
Functions that accept a `scanf(3)`-compliant format string should be labeled `H5_ATTR_FORMAT(scanf,format_argno,variadic_argno)`.
##  Functions that do never return
The definition of a function that always causes the program to abort and hang should be labeled `H5_ATTR_NORETURN` to help the compiler see which flows of control are infeasible.
##  Other attributes
TBD**
##  Unused variables and parameters
Compilers will warn about unused parameters and variables—developers should pay attention to those warnings and make an effort to prevent them.
Some function parameters and variables are unused in **all** configurations of the project. Ordinarily, such parameters and variables should be deleted. However, sometimes it is possible to foresee a parameter being used, or removing it would change an API, or a parameter has to be defined to conform a function to some function pointer type. In those cases, it's permissible to mark a symbol `H5_ATTR_UNUSED`.
Other parameters and variables are unused in **some** configurations of the project, but not all. A symbol may fall into disuse in some configuration in the future—then the compiler should warn, and the symbol should not be defined—so developers should try to label a sometimes-unused symbol with an attribute that's specific to the configurations where the symbol is (or is not) expected to be used. The library provides the following attributes for that purpose: 
  * `H5_ATTR_DEPRECATED_USED`: used only if deprecated symbols **are** enabled 
  * `H5_ATTR_NDEBUG_UNUSED`: used only if `NDEBUG` is **not** #defined 
  * `H5_ATTR_DEBUG_API_USED`: used if the debug API **is** enabled 
  * `H5_ATTR_PARALLEL_UNUSED`: used only if Parallel HDF5 is **not** configured 
  * `H5_ATTR_PARALLEL_USED`: used only if Parallel HDF5 **is** configured


Some attributes may be phased in or phased out in the future.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


