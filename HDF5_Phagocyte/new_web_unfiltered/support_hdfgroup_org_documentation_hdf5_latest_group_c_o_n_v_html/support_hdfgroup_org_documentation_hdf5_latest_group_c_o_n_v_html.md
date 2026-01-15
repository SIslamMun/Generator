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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title2 "H2")
  * [◆ H5Tcompiler_conv()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title3 "H2")
  * [◆ H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title4 "H2")
  * [◆ H5Tfind()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title5 "H2")
  * [◆ H5Tregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title6 "H2")
  * [◆ H5Tunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#func-members)
Conversion Function
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
## Detailed Description
##  Functions  
---  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Tcompiler_conv](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga13fc42ad63ffd1e40e6672d30c8dd1cf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id)  
| Check whether the library's default conversion is hard conversion.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tconvert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id, size_t nelmts, void *buf, void *background, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Converts data from one specified datatype to another.   
  
[H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) |  [H5Tfind](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga71940c1637a309748fe93b6dceabd02f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id, [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html) **pcdata)  
| Finds a conversion function.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a) ([H5T_pers_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7e) pers, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id, [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) func)  
| Registers a datatype conversion function.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#gacc791af473dd1de512dacf0e8d6554f1) ([H5T_pers_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7e) pers, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id, [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) func)  
| Removes a conversion function.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga13fc42ad63ffd1e40e6672d30c8dd1cf)H5Tcompiler_conv()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Tcompiler_conv  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_id_ )  
Check whether the library's default conversion is hard conversion.  

Parameters
     [in] | src_id | Datatype identifier of source datatype   
---|---|---  
[in] | dst_id | Datatype identifier of destination datatype 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5Tcompiler_conv()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga13fc42ad63ffd1e40e6672d30c8dd1cf "Check whether the library's default conversion is hard conversion.") determines whether the library's conversion function from type `src_id` to type `dst_id` is a compiler (hard) conversion or not. A compiler conversion uses compiler's casting; a library (soft) conversion uses the library's own conversion function. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a)H5Tconvert()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tconvert  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_id_ ,   
|  | size_t |  _nelmts_ ,   
|  | void * |  _buf_ ,   
|  | void * |  _background_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ )  
Converts data from one specified datatype to another. 
* * * 

Parameters
     [in] | src_id | Datatype identifier of source datatype   
---|---|---  
[in] | dst_id | Datatype identifier of destination datatype   
[in] | nelmts | Size of array `buf`  
[in,out] | buf | Array containing pre- and post-conversion values   
[in] | background | Optional background buffer   
[in] | plist_id | Dataset transfer property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") converts `nelmts` elements from a source datatype, specified by `src_id`, to a destination datatype, `dst_id`. The source elements are packed in `buf` and on return the destination elements will be packed in `buf`. That is, the conversion is performed in place.
The optional background buffer is for use with compound datatypes. It is an array of `nelmts` values for the destination datatype which can then be merged with the converted values to recreate the compound datatype. For instance, background might be an array of structs with the `a` and `b` fields already initialized and the conversion of buf supplies the `c` and `d` field values.
The parameter `plist_id` contains the dataset transfer property list identifier which is passed to the conversion functions. As of Release 1.2, this parameter is only used to pass along the variable-length datatype custom allocation information. 

Note
     [H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") will not resize the buffer `buf`; it must be large enough to hold the larger of the input and output data. 

Version
    1.6.3 `nelmts` parameter type changed to size_t.       1.4.0 `nelmts` parameter type changed to [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga71940c1637a309748fe93b6dceabd02f)H5Tfind()
[H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) H5Tfind  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_id_ ,   
|  |  [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html) ** |  _pcdata_ )  
Finds a conversion function.  

Parameters
     [in] | src_id | Datatype identifier of source datatype   
---|---|---  
[in] | dst_id | Datatype identifier of destination datatype   
[out] | pcdata | Pointer to type conversion data 

Returns
    Returns a pointer to a suitable conversion function if successful. Otherwise returns NULL.  
[H5Tfind()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga71940c1637a309748fe93b6dceabd02f "Finds a conversion function.") finds a conversion function that can handle a conversion from type `src_id` to type `dst_id`. The `pcdata` argument is a pointer to a pointer to type conversion data which was created and initialized by the soft type conversion function of this path when the conversion function was installed on the path. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a)H5Tregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tregister  | ( | [H5T_pers_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7e) |  _pers_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_id_ ,   
|  | [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) |  _func_ )  
Registers a datatype conversion function.  

Parameters
     [in] | pers | Conversion function type   
---|---|---  
[in] | name | Name displayed in diagnostic output   
[in] | src_id | Datatype identifier of source datatype   
[in] | dst_id | Datatype identifier of destination datatype   
[in] | func | Function to convert between source and destination datatypes 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a "Registers a datatype conversion function.") registers a hard or soft conversion function for a datatype conversion path. The parameter `pers` indicates whether a conversion function is hard ([H5T_PERS_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7eaa4f363fa661ce571b050873e7e653b98)) or soft ([H5T_PERS_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7ea9f5fe7567029ac797b118d4ef16f206c)). User-defined functions employing compiler casting are designated as _hard_ ; other user-defined conversion functions registered with the HDF5 library (with [H5Tregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a "Registers a datatype conversion function.") ) are designated as _soft_. The HDF5 library also has its own hard and soft conversion functions.
A conversion path can have only one hard function. When type is [H5T_PERS_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7eaa4f363fa661ce571b050873e7e653b98), `func` replaces any previous hard function.
When type is [H5T_PERS_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7ea9f5fe7567029ac797b118d4ef16f206c), [H5Tregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a "Registers a datatype conversion function.") adds the function to the end of the master soft list and replaces the soft function in all applicable existing conversion paths. Soft functions are used when determining which conversion function is appropriate for this path.
The `name` is used only for debugging and should be a short identifier for the function.
The path is specified by the source and destination datatypes `src_id` and `dst_id`. For soft conversion functions, only the class of these types is important.
The type of the conversion function pointer is declared as: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_id, [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html) *cdata, size_t nelmts, size_t buf_stride,
size_t bkg_stride, void *buf, void *bkg, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_xfer_plist);
The [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html) `struct` is declared as: 
typedef struct [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html) {
[H5T_cmd_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a9bde6125943ed5565062a4c12c7be8bd) [command](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html#a5de0a2b4f94c462798b1463909866a89); 
[H5T_bkg_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a6d9a04bf7a35625abc25f1bae32c8334) [need_bkg](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html#a6d67d8363314e88174fcb11755b84f2e); 
bool [recalc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html#abb6dd63884a1fb0c4e53387168646dde); 
void *[priv](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html#a8b6505c37d4ff95854b8b00527e4d9fa); 
} [H5T_cdata_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_t__cdata__t.html); 

Since
    1.6.3 The following change occurred in the [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) function: the `nelmts` parameter type changed to size_t. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#gacc791af473dd1de512dacf0e8d6554f1)H5Tunregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tunregister  | ( | [H5T_pers_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#ade8bfa5625e0b17bae72f10246be3c7e) |  _pers_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_id_ ,   
|  | [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) |  _func_ )  
Removes a conversion function.  

Parameters
     [in] | pers | Conversion function type   
---|---|---  
[in] | name | Name displayed in diagnostic output   
[in] | src_id | Datatype identifier of source datatype   
[in] | dst_id | Datatype identifier of destination datatype   
[in] | func | Function to convert between source and destination datatypes 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#gacc791af473dd1de512dacf0e8d6554f1 "Removes a conversion function.") removes a conversion function matching criteria such as soft or hard conversion, source and destination types, and the conversion function.
If a user is trying to remove a conversion function he registered, all parameters can be used. If he is trying to remove a library's default conversion function, there is no guarantee the `name` and `func` parameters will match the user's chosen values. Passing in some values may cause this function to fail. A good practice is to pass in NULL as their values.
All parameters are optional. The missing parameters will be used to generalize the search criteria.
The conversion function pointer type declaration is described in [H5Tregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga0a23a3cb9f24bd79fae7d2d8c412a25a "Registers a datatype conversion function."). 

Version
    1.6.3 The following change occurred in the [H5T_conv_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tdevelop_8h.html#a5dcae1e3122cc65cb9553ce72d9ddc54) function: the `nelmts` parameter type changed to size_t. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


