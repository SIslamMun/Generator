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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title0 "H2")
  * [ Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title1 "H2")
  * [ Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title2 "H2")
  * [Function/Subroutine Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title3 "H2")
  * [◆ h5dsattach_scale_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title4 "H2")
  * [◆ h5dsdetach_scale_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title5 "H2")
  * [◆ h5dsget_label_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title6 "H2")
  * [◆ h5dsget_num_scales_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title7 "H2")
  * [◆ h5dsget_scale_name_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title8 "H2")
  * [◆ h5dsis_attached_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title9 "H2")
  * [◆ h5dsis_scale_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title10 "H2")
  * [◆ h5dsset_label_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title11 "H2")
  * [◆ h5dsset_scale_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#title12 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#namespaces) | [Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#func-members)
Fortran High Level Dimension Scales (H5DS) Interface
## Detailed Description 

See also
     [HDF5 Dimension Scales APIs (H5DS)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html), C-HL API      [HDF5 High Level Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html), User Guide 
##  Modules  
---  
module  | [h5ds](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5ds.html)  
| This module contains Fortran interfaces for H5DS.   
  
##  Functions/Subroutines  
---  
subroutine  |  [h5dsattach_scale_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga50588af3875bff680ae903a55a3f0cdf) (did, dsid, idx, errcode)  
| Attach dimension scale dsid to dimension `idx` of dataset `did`.   
  
subroutine  |  [h5dsdetach_scale_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga15843c184487bf78bd73eef0c4693c5e) (did, dsid, idx, errcode)  
| Detach dimension scale dsid from the dimension idx of dataset `did`.   
  
subroutine  |  [h5dsget_label_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga12ed0456a8ac947465dd2a427eff6917) (did, idx, label, size, errcode)  
| Read the `label` for dimension `idx` of `did` into buffer `label`.   
  
subroutine  |  [h5dsget_num_scales_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga229397f9f94eb63f1491863a3042cdc4) (did, idx, num_scales, errcode)  
| Determines how many Dimension Scales are attached to dimension idx of `did`.   
  
subroutine  |  [h5dsget_scale_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#gaeb9f29853c60e6ee68846cfc22167c3b) (did, name, size, errcode)  
| Read the name of scale `did` into buffer name.   
  
subroutine  |  [h5dsis_attached_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga19cad5f337c36e2ac26bd5f3809e6567) (did, dsid, idx, is_attached, errcode)  
| Report if dimension scale dsid is currently attached to dimension idx of dataset did.   
  
subroutine  |  [h5dsis_scale_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga6640937911372482b872e03b80fbad88) (did, is_scale, errcode)  
| Determines whether `did` is a Dimension Scale.   
  
subroutine  |  [h5dsset_label_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga7f9cb5b5f8f5eb1e1b3c069038a1674d) (did, idx, label, errcode)  
| Set label for the dimension `idx` of `did` to the value `label`.   
  
subroutine  |  [h5dsset_scale_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#gaeb5d3e43c9934bb2fe7e3dbdec534417) (dsid, errcode, dimname)  
| Convert dataset `dsid` to a dimension scale, with optional name, `dimname`.   
  
## Function/Subroutine Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga50588af3875bff680ae903a55a3f0cdf)h5dsattach_scale_f()
subroutine h5dsattach_scale_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _dsid_ ,   
|  | integer, intent(in) |  _idx_ ,   
|  | integer |  _errcode_ )  
Attach dimension scale dsid to dimension `idx` of dataset `did`.  

Parameters
     did | The dataset.   
---|---  
dsid | The scale to be attached.   
idx | The dimension of `did` that `dsid` is associated with.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSattach_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga15843c184487bf78bd73eef0c4693c5e)h5dsdetach_scale_f()
subroutine h5dsdetach_scale_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _dsid_ ,   
|  | integer, intent(in) |  _idx_ ,   
|  | integer |  _errcode_ )  
Detach dimension scale dsid from the dimension idx of dataset `did`.  

Parameters
     did | The dataset.   
---|---  
dsid | The scale to be detached.   
idx | The dimension of `did` to detach.   
errcode | Returns 0 if successful and -1 if it fails.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga12ed0456a8ac947465dd2a427eff6917)h5dsget_label_f()
subroutine h5dsget_label_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer, intent(in) |  _idx_ ,   
|  | character(len=*), intent(inout) |  _label_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(inout) |  _size_ ,   
|  | integer |  _errcode_ )  
Read the `label` for dimension `idx` of `did` into buffer `label`.  

Parameters
     did | The dataset.   
---|---  
idx | The dimension.   
label | The label.   
size | The length of the `label` buffer.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSget_label()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga265aa3dadb5d067c112620ffe01f96d7)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga229397f9f94eb63f1491863a3042cdc4)h5dsget_num_scales_f()
subroutine h5dsget_num_scales_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer, intent(in) |  _idx_ ,   
|  | integer, intent(inout) |  _num_scales_ ,   
|  | integer |  _errcode_ )  
Determines how many Dimension Scales are attached to dimension idx of `did`.  

Parameters
     did | The dataset to query.   
---|---  
idx | The dimension of `did` to query.   
num_scales | Number of Dimension Scales associated with `did`.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSget_num_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga562b91296f739a1833a1e1cfc64c7726)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#gaeb9f29853c60e6ee68846cfc22167c3b)h5dsget_scale_name_f()
subroutine h5dsget_scale_name_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | character(len=*), intent(inout) |  _name_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(inout) |  _size_ ,   
|  | integer |  _errcode_ )  
Read the name of scale `did` into buffer name.  

Parameters
     did | Dimension scale identifier.   
---|---  
name | Buffer to contain the returned name.   
size | Size in bytes, of the name buffer.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSget_scale_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga19cad5f337c36e2ac26bd5f3809e6567)h5dsis_attached_f()
subroutine h5dsis_attached_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _dsid_ ,   
|  | integer, intent(in) |  _idx_ ,   
|  | logical, intent(out) |  _is_attached_ ,   
|  | integer |  _errcode_ )  
Report if dimension scale dsid is currently attached to dimension idx of dataset did.  

Parameters
     did | The dataset.   
---|---  
dsid | The scale to be attached.   
idx | The dimension of `did` that `dsid` is associated with.   
is_attached | If dimension scale `dsid` is currently attached to dimension `idx` of dataset `did`.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSis_attached()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad7bbd73a68b031f6ac4c305333b0c8b7)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga6640937911372482b872e03b80fbad88)h5dsis_scale_f()
subroutine h5dsis_scale_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | logical, intent(out) |  _is_scale_ ,   
|  | integer, intent(out) |  _errcode_ )  
Determines whether `did` is a Dimension Scale.  

Parameters
     did | The data set to query.   
---|---  
is_scale | If is a Dimension Scale.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSis_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#ga7f9cb5b5f8f5eb1e1b3c069038a1674d)h5dsset_label_f()
subroutine h5dsset_label_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _did_ ,   
---|---|---|---  
|  | integer, intent(in) |  _idx_ ,   
|  | character(len=*), intent(in) |  _label_ ,   
|  | integer |  _errcode_ )  
Set label for the dimension `idx` of `did` to the value `label`.  

Parameters
     did | The data set.   
---|---  
idx | The dimension.   
label | The label.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5DSset_label()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html#gaeb5d3e43c9934bb2fe7e3dbdec534417)h5dsset_scale_f()
subroutine h5dsset_scale_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _dsid_ ,   
---|---|---|---  
|  | integer |  _errcode_ ,   
|  | character(len=*), intent(in), optional |  _dimname_ )  
Convert dataset `dsid` to a dimension scale, with optional name, `dimname`.  

Parameters
     dsid | The dataset to be made a Dimemsion Scale.   
---|---  
errcode | Returns 0 if successful and -1 if it fails.   
dimname | The dimension name  
See C API: [H5DSset_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


