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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title1 "H2")
  * [ Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title2 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title3 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title4 "H2")
  * [◆ H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title5 "H2")
  * [Typedef Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title6 "H2")
  * [◆ H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title7 "H2")
  * [◆ H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title8 "H2")
  * [◆ H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#define-members) | [Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#typedef-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#func-members)
H5Zdevelop.h File Reference
`#include "H5Zpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5Z_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__cb__t.html)  
struct  | [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html)  
struct  | [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html)  
##  Macros  
---  
#define  |  [H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9) (1)  
##  Typedefs  
---  
typedef [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)(*  |  [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id)  
| This callback determines if a filter can be applied to the dataset with the characteristics provided.   
  
typedef size_t(*  |  [H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6)) (unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t *buf_size, void **buf)  
| The filter operation callback function, defining a filter's operation on data.   
  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id)  
| The filter operation callback function, defining a filter's operation on data.   
  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b) (const void *cls)  
| Registers a new filter with the HDF5 library.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8) ([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) id)  
| Unregisters a filter.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9)H5Z_CLASS_T_VERS
#define H5Z_CLASS_T_VERS (1)  
---  
Current version of the H5Z_class_t struct 
## Typedef Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510)H5Z_can_apply_func_t
typedef [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)(* H5Z_can_apply_func_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id)  
---  
This callback determines if a filter can be applied to the dataset with the characteristics provided.  

Parameters
     [in] | dcpl_id | Dataset creation property list identifier   
---|---|---  
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
Before a dataset gets created, the [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) callbacks for any filters used in the dataset creation property list are called with the dataset's dataset creation property list, the dataset's datatype and a dataspace describing a chunk (for chunked dataset storage).
The [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) callback must determine if the combination of the dataset creation property list setting, the datatype and the dataspace represent a valid combination to apply this filter to. For example, some cases of invalid combinations may involve the filter not operating correctly on certain datatypes (or certain datatype sizes), or certain sizes of the chunk dataspace.
The [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) callback can be the NULL pointer, in which case, the library will assume that it can apply to any combination of dataset creation property list values, datatypes and dataspaces.
The [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) callback returns positive a valid combination, zero for an invalid combination and negative for an error. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6)H5Z_func_t
typedef size_t(* H5Z_func_t) (unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t *buf_size, void **buf)  
---  
The filter operation callback function, defining a filter's operation on data.  

Parameters
     [in] | flags | Bit vector specifying certain general properties of the filter   
---|---|---  
[in] | cd_nelmts | Number of elements in `cd_values`  
[in] | cd_values | Auxiliary data for the filter   
[in] | nbytes | The number of valid bytes in `buf` to be filtered   
[in,out] | buf_size | The size of `buf`  
[in,out] | buf | The filter buffer 

Returns
    Returns the number of valid bytes of data contained in `buf`. In the case of failure, the return value is 0 (zero) and all pointer arguments are left unchanged.  
A filter gets definition flags and invocation flags (defined above), the client data array and size defined when the filter was added to the pipeline, the size in bytes of the data on which to operate, and pointers to a buffer and its allocated size.
The filter should store the result in the supplied buffer if possible, otherwise it can allocate a new buffer, freeing the original. The allocated size of the new buffer should be returned through the `buf_size` pointer and the new buffer through the `buf` pointer.
The return value from the filter is the number of bytes in the output buffer. If an error occurs then the function should return zero and leave all pointer arguments unchanged. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e)H5Z_set_local_func_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5Z_set_local_func_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id)  
---  
The filter operation callback function, defining a filter's operation on data.  

Parameters
     [in] | dcpl_id | Dataset creation property list identifier   
---|---|---  
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
After the [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) callbacks are checked for new datasets, the [H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) callbacks for any filters used in the dataset creation property list are called. These callbacks receive the dataset's private copy of the dataset creation property list passed in to [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (i.e. not the actual property list passed in to [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)) and the datatype ID passed in to [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (which is not copied and should not be modified) and a dataspace describing the chunk (for chunked dataset storage) (which should also not be modified).
The [H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) callback must set any parameters that are specific to this dataset, based on the combination of the dataset creation property list values, the datatype and the dataspace. For example, some filters perform different actions based on different datatypes (or datatype sizes) or different number of dimensions or dataspace sizes.
The [H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) callback can be the NULL pointer, in which case, the library will assume that there are no dataset-specific settings for this filter.
The [H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) callback must return non-negative on success and negative for an error. 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5Zdevelop.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


