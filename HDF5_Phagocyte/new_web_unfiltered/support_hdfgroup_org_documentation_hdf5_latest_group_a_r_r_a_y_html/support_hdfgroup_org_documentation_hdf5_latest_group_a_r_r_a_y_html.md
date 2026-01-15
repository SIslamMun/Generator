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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title3 "H2")
  * [◆ H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title4 "H2")
  * [◆ H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title5 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title6 "H2")
  * [◆ H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title7 "H2")
  * [◆ H5Tarray_create2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title8 "H2")
  * [◆ H5Tget_array_dims1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title9 "H2")
  * [◆ H5Tget_array_dims2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title10 "H2")
  * [◆ H5Tget_array_ndims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#func-members)
Array Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
## Detailed Description
##  Macros  
---  
#define  |  [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618)  
#define  |  [H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) [H5Tget_array_dims2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769)  
##  Functions  
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Tarray_create1](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id, int ndims, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dim[], const int perm[])  
| Creates an array datatype object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id, unsigned ndims, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dim[])  
| Creates an array datatype object.   
  
int  |  [H5Tget_array_dims1](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga40dca4c9bdc5e6781a07830570a476ca) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims[], int perm[])  
| Retrieves sizes of array dimensions.   
  
int  |  [H5Tget_array_dims2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims[])  
| Retrieves sizes of array dimensions.   
  
int  |  [H5Tget_array_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gadec89de23da8efaba4677abfd818a9c0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id)  
| Returns the rank of an array datatype.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96)H5Tarray_create
#define H5Tarray_create [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618)  
---  
[H5Tarray_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) is a macro that is mapped to either [H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80 "Creates an array datatype object.") or [H5Tarray_create2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618 "Creates an array datatype object.").  


See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473)H5Tget_array_dims
#define H5Tget_array_dims [H5Tget_array_dims2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769)  
---  
[H5Tget_array_dims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) is a macro that is mapped to either [H5Tget_array_dims1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga40dca4c9bdc5e6781a07830570a476ca "Retrieves sizes of array dimensions.") or [H5Tget_array_dims2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769 "Retrieves sizes of array dimensions.").  


See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80)H5Tarray_create1()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Tarray_create1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _base_id_ ,   
---|---|---|---  
|  | int |  _ndims_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _dim_[],   
|  | const int |  _perm_[] )  
Creates an array datatype object.  

Parameters
     [in] | base_id | Datatype identifier for the array base datatype   
---|---|---  
[in] | ndims | Rank of the array   
[in] | dim | Size of each array dimension   
[in] | perm | Dimension permutation (Currently not implemented.) 

Returns
    Returns a array datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000099)**
    This function has been renamed from [H5Tarray_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) and is deprecated in favor of the macro [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) or the function [H5Tarray_create2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618 "Creates an array datatype object.").  
[H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80 "Creates an array datatype object.") creates a new array datatype object.  
  
`base_id` is the datatype of every element of the array, i.e., of the number at each position in the array.
`rank` is the number of dimensions and the size of each dimension is specified in the array dims. The value of rank is currently limited to [H5S_MAX_RANK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a265cb2343f05cb71831119c90de31a8f) and must be greater than 0 (zero). All dimension sizes specified in dims must be greater than 0 (zero).
The array `perm` is designed to contain the dimension permutation, i.e. C versus FORTRAN array order. (The parameter perm is currently unused and is not yet implemented.) 

Version
    1.8.0 Function [H5Tarray_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) renamed to [H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80 "Creates an array datatype object.") and deprecated in this release. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618)H5Tarray_create2()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Tarray_create2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _base_id_ ,   
---|---|---|---  
|  | unsigned |  _ndims_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _dim_[] )  
Creates an array datatype object.  

Parameters
     [in] | base_id | Datatype identifier for the array base datatype   
---|---|---  
[in] | ndims | Rank of the array   
[in] | dim | Size of each array dimension 

Returns
    Returns a array datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Tarray_create2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618 "Creates an array datatype object.") creates a new array datatype object.  
  
`base_id` is the datatype of every element of the array, i.e., of the number at each position in the array.
`ndims` is the number of dimensions and the size of each dimension is specified in the array `dim`. The value of `rank` is currently limited to [H5S_MAX_RANK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a265cb2343f05cb71831119c90de31a8f) and must be greater than 0 (zero). All dimension sizes specified in `dim` must be greater than 0 (zero). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga40dca4c9bdc5e6781a07830570a476ca)H5Tget_array_dims1()
int H5Tget_array_dims1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _dims_[],   
|  | int |  _perm_[] )  
Retrieves sizes of array dimensions.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[out] | dims | Sizes of array dimensions   
[out] | perm | Dimension permutations (This parameter is not used.) 

Returns
    Returns the non-negative number of dimensions of the array type if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000100)**
    This function has been renamed from [H5Tget_array_dims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) and is deprecated in favor of the macro [H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) or the function [H5Tget_array_dims2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769 "Retrieves sizes of array dimensions.").  
[H5Tget_array_dims1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga40dca4c9bdc5e6781a07830570a476ca "Retrieves sizes of array dimensions.") returns the sizes of the dimensions and the dimension permutations of the specified array datatype object.
The sizes of the dimensions are returned in the array `dims`. 

Version
    1.8.0 Function [H5Tarray_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) renamed to [H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80 "Creates an array datatype object.") and deprecated in this release. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769)H5Tget_array_dims2()
int H5Tget_array_dims2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _dims_[] )  
Retrieves sizes of array dimensions.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[out] | dims | Sizes of array dimensions 

Returns
    Returns the non-negative number of dimensions of the array type if successful; otherwise returns a negative value.  
[H5Tget_array_dims2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769 "Retrieves sizes of array dimensions.") returns the sizes of the dimensions of the specified array datatype object in the array `dims`. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gadec89de23da8efaba4677abfd818a9c0)H5Tget_array_ndims()
int H5Tget_array_ndims  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _type_id_ | ) |   
---|---|---|---|---|---  
Returns the rank of an array datatype.  

Parameters
     [in] | type_id | Datatype identifier  
---|---|--- 

Returns
    Returns the rank of the array if successful; otherwise returns a negative value.  
[H5Tget_array_ndims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gadec89de23da8efaba4677abfd818a9c0 "Returns the rank of an array datatype.") returns the rank, i.e., the number of dimensions, of an array datatype object. 

Since
    1.2.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


