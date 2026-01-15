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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title2 "H2")
  * [◆ image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title3 "H2")
  * [◆ image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title4 "H2")
  * [◆ image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title5 "H2")
  * [◆ image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title6 "H2")
  * [◆ udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title7 "H2")
  * [◆ udata_copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title8 "H2")
  * [◆ udata_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#pub-attribs)
H5FD_file_image_callbacks_t Struct Reference
`#include <src/H5FDpublic.h>`
## Detailed Description
Define structure to hold file image callbacks 
##  Data Fields  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) )(void *ptr, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
void *(*  |  [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32) )(size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
void *(*  |  [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030) )(void *dest, const void *src, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
void *(*  |  [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d) )(void *ptr, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
void *  | [udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed)  
| The final field in the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) struct, provides a pointer to user-defined data. This pointer will be passed to the image_malloc, image_memcpy, image_realloc, and image_free callbacks. Define udata as NULL if no user-defined data is provided.   
  
void *(*  |  [udata_copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a20cdfcd242ef041c748f47f8de39774d) )(void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [udata_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a72907de159ae0b35408bbfe75f7c3359) )(void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890)image_free
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* image_free) (void *ptr, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | ptr | Pointer to the buffer being reallocated   
---|---|---  
[in] | file_image_op | A value from [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) indicating the operation being performed on the file image when this callback is invoked   
[in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32)image_malloc
void *(* image_malloc) (size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | size | Size in bytes of the file image buffer to allocate   
---|---|---  
[in] | file_image_op | A value from [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) indicating the operation being performed on the file image when this callback is invoked   
[in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030)image_memcpy
void *(* image_memcpy) (void *dest, const void *src, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | dest | Address of the destination buffer   
---|---|---  
[in] | src | Address of the source buffer   
[in] | size | Size in bytes of the file image buffer to allocate   
[in] | file_image_op | A value from [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) indicating the operation being performed on the file image when this callback is invoked   
[in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d)image_realloc
void *(* image_realloc) (void *ptr, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | ptr | Pointer to the buffer being reallocated   
---|---|---  
[in] | size | Size in bytes of the file image buffer to allocate   
[in] | file_image_op | A value from [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) indicating the operation being performed on the file image when this callback is invoked   
[in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed)udata
void* udata  
---  
The final field in the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) struct, provides a pointer to user-defined data. This pointer will be passed to the image_malloc, image_memcpy, image_realloc, and image_free callbacks. Define udata as NULL if no user-defined data is provided. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a20cdfcd242ef041c748f47f8de39774d)udata_copy
void *(* udata_copy) (void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
---|---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a72907de159ae0b35408bbfe75f7c3359)udata_free
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* udata_free) (void *[udata](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a697ce711b67313990d351b5c95f87aed))  
--- 

Parameters
     [in] | udata | Value passed in in the H5Pset_file_image_callbacks parameter `udata`  
---|---|---  
* * *
The documentation for this struct was generated from the following file:
  * src/[H5FDpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)


  * [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


