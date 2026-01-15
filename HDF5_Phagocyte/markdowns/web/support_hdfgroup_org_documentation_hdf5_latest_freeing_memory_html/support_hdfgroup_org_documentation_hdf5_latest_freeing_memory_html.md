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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title0 "H1")
  * [ The Windows C Run-time (CRT)](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title1 "H1")
  * [ Affected API Calls](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title2 "H1")
  * [ Mitigation](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title3 "H1")
  * [ Use the Same Memory Manager/Correct C Run‐time Everywhere](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title4 "H2")
  * [ Use the H5free_memory Function](https://support.hdfgroup.org/documentation/hdf5/latest/freeing_memory.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Freeing Memory Allocated by the HDF5 Library
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
* * *
Several functions in the HDF5 C API return buffers allocated by the HDF5 Library. When application code uses a different library for memory management than the HDF Library, a corrupt heap or a resource leak can occur when these allocated buffers are freed. This is most commonly a problem on Windows systems since Microsoft implements C library functions in Visual Studio-­­specific libraries which do not share heap state.
Introduced with HDF5 Release 1.8.13 May 15, 2014
This document describes this problem and the steps users can take to mitigate the problem. This document also introduces the new [H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") function.
#  Introduction
In the HDF5 Library, responsibility for the allocation and freeing of memory is usually the responsibility of the same component: either the library or the user's code. When data that would normally be stored in dynamically­allocated memory must be returned from the library, the user is usually asked to allocate a buffer which is passed to the function and then filled by the library. The complication is that the user must be able to determine the buffer's size. The mechanism for this is for the user to make a preliminary call, passing a NULL pointer in for the buffer. The function will then return the appropriate number of bytes for the user to allocate. See the example below.
_Example1. Determining the buffer size with a preliminary call_
ssize_t size;
size_t bufsize;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) object_id;
char *comment;
…
size = [H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b)(object_id, NULL, bufsize); // determine size
bufsize = size;
comment = (char *)malloc(bufsize * sizeof(char));
size = [H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b)(object_id, comment, bufsize); // fill buffer
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b)
ssize_t H5Oget_comment(hid_t obj_id, char *comment, size_t bufsize)
Retrieves comment for specified object.
There are, however, several API calls in which the buffer is allocated by the HDF5 Library and returned to the user who is responsible for freeing it. This can be a problem when memory in the application and HDF5 Library are managed via different libraries as it can result in resource leaks or a corrupted heap. This heap corruption can result in subtle bugs that can be very difficult to reproduce and diagnose. In most cases, having the library allocate memory and the application free it is not a problem since memory operations will resolve down to the operating system's memory manager; however, there are cases where this is not true. For example, a debug memory manager may be in use by the application code but not the library. A complication that is unique to Windows is that the C standard library functions are implemented in Visual­Studio­specific C run-time (CRT) libraries. When different versions of Visual Studio are used to compile the library and application code, the allocate and free calls are made in different libraries, which do not share state, leading to the previously mentioned resource and corruption issues.
#  The Windows C Run-time (CRT)
Microsoft implements the standard C library functions in debug and release libraries that are specific to each version of Visual Studio1. Each library is a separate entity and maintains its own internal CRT object state, file handles, and heap information. Creating an object in one CRT and destroying it in another CRT may appear to work but can cause corruption of one CRT and resource leaks in the other.
![](https://support.hdfgroup.org/documentation/hdf5/latest/FreeingMemory_fig1.png)  
---  
These problems are normally avoided on Windows by ensuring that all components that can return CRT resources are linked to the same CRT dynamic link library (DLL). Unfortunately, even debug and release CRTs are housed in separate DLLs, so this is not an easy solution to implement. Using static linkage does not avoid this problem since separate copies of the CRT are created in each statically linked component.
  * 1 The names of these libraries are of the form MSVCR<#>.dll, where <#> is the Visual Studio version. For example, MSVCR110.dll corresponds to Visual Studio 11.0 (2012).


#  Affected API Calls
This is a list of the API calls that are affected. 
  * [H5Eget_major](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga91fa7fb56da6f08f9c293a6ce89c7819 "Returns a character string describing an error specified by a major error number.")
  * [H5Eget_minor](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4975325db13bc5cf44d72d4ef0394034 "Returns a character string describing an error specified by a minor error number.")
  * [H5Pget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82 "Retrieves the name of a class.")
  * [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66 "Retrieves the name of a compound or enumeration datatype member.")
  * [H5Tget_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#gaf1d0f634ac1a3b4220b8fe0197b93832 "Gets the tag associated with an opaque datatype.")


#  Mitigation
There are several potential solutions to the problem of freeing memory allocated by the HDF5 Library.
##  Use the Same Memory Manager/Correct C Run‐time Everywhere
Both application code and the HDF5 Library must use the same memory allocator. When using Visual Studio, both the Visual Studio version and release/debug state must be identical. As of HDF5 1.8.12, this is the only available solution.
##  Use the H5free_memory Function
A new function called [H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") has been created and is essentially a thin wrapper for the run­ time's free() call. This function would be used to free any memory allocated by the library. This solution has the advantages of being extremely easy to implement and intuitive to use. It can also be used as a solution with legacy API calls, so it would be necessary even if we modify the HDF5 API. This function will also be extremely useful when HDF5 is wrapped for use with managed languages such as Java, .NET, and Python so that the wrappers can properly clean up resources.
See the [H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") entry in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) for more information.
Note that the creation of this function does not imply that it will be acceptable for new API calls to be created that return library­allocated memory. The preferred mechanism will still be to use the "preliminary call" scheme described in the "Introduction" on page 4 where the user allocates the buffer.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


