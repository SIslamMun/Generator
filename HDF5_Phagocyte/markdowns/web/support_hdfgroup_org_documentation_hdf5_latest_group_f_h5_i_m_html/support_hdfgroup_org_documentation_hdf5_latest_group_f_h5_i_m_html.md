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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title0 "H2")
  * [ Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title1 "H2")
  * [ Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title2 "H2")
  * [Function/Subroutine Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title3 "H2")
  * [◆ h5imget_image_info_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title4 "H2")
  * [◆ h5imget_npalettes_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title5 "H2")
  * [◆ h5imget_palette_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title6 "H2")
  * [◆ h5imget_palette_info_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title7 "H2")
  * [◆ h5imis_image_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title8 "H2")
  * [◆ h5imis_palette_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title9 "H2")
  * [◆ h5imlink_palette_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title10 "H2")
  * [◆ h5immake_image_24bit_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title11 "H2")
  * [◆ h5immake_image_8bit_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title12 "H2")
  * [◆ h5immake_palette_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title13 "H2")
  * [◆ h5imread_image_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title14 "H2")
  * [◆ h5imunlink_palette_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#title15 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#namespaces) | [Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#func-members)
Fortran High Level Images (H5IM) Interface
## Detailed Description 

See also
     [HDF5 Images API (H5IM)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html), C-HL API      [HDF5 High Level Images](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i_m__u_g.html), User Guide 
##  Modules  
---  
module  | [h5im](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5im.html)  
| This module contains Fortran interfaces for H5IM.   
  
##  Functions/Subroutines  
---  
subroutine  |  [h5imget_image_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga37655210991f287c570e619cfedc28bf) (loc_id, dset_name, width, height, planes, interlace, npals, errcode)  
| Gets information about an image dataset (dimensions, interlace mode and number of associated palettes).   
  
subroutine  |  [h5imget_npalettes_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gac06e374ebd82535e70fd14d214dc3f32) (loc_id, image_name, npals, errcode)  
| Gets the number of palettes associated to an image.   
  
subroutine  |  [h5imget_palette_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga2069e82b8d26612f7e261ec90f7b45a2) (loc_id, image_name, pal_number, pal_data, errcode)  
| Gets the palette dataset.   
  
subroutine  |  [h5imget_palette_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga84c037ebf29227878e876740156fe666) (loc_id, image_name, pal_number, pal_dims, errcode)  
| Gets information about a palette dataset (dimensions).   
  
integer function  |  [h5imis_image_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga96ac5556029f5723899d3ed55e7cc419) (loc_id, dset_name)  
| Inquires if a dataset is an image.   
  
integer function  |  [h5imis_palette_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga9ac7575820d1b75c6661a78107f2eec5) (loc_id, dset_name)  
| Inquires if a dataset is a palette. Returns zero (false), a positive (true) or a negative (failure) value.   
  
subroutine  |  [h5imlink_palette_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga0986868cd2ddef408ac55d704b064a60) (loc_id, image_name, pal_name, errcode)  
| This function attaches a palette to an existing image dataset.   
  
subroutine  |  [h5immake_image_24bit_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gae17a8216ab5725dc684ed7baea996680) (loc_id, dset_name, width, height, il, buf, errcode)  
| Creates and writes an image a 24 bit image.   
  
subroutine  |  [h5immake_image_8bit_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gac30cbab36df84de283e30d9de9bdc848) (loc_id, dset_name, width, height, buf, errcode)  
| Creates and writes an image an 8 bit image.   
  
subroutine  |  [h5immake_palette_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga8c66f1723f02ab514631b1ca448af532) (loc_id, pal_name, pal_dims, pal_data, errcode)  
| Creates and writes a palette.   
  
subroutine  |  [h5imread_image_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gace5754228364bcde3e21d6d1cdb0cb5e) (loc_id, dset_name, buf, errcode)  
| Reads image data from disk.   
  
subroutine  |  [h5imunlink_palette_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga6dfdaf798d1f36076aea143c1f23f8c2) (loc_id, image_name, pal_name, errcode)  
| This function detaches a palette to an existing image dataset.   
  
## Function/Subroutine Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga37655210991f287c570e619cfedc28bf)h5imget_image_info_f()
subroutine h5imget_image_info_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _width_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _height_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _planes_ ,   
|  | character(len=*), intent(inout) |  _interlace_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _npals_ ,   
|  | integer |  _errcode_ )  
Gets information about an image dataset (dimensions, interlace mode and number of associated palettes).  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset.   
width | The width of the image.   
height | The height of the image.   
planes | The number of color planes of the image.   
interlace | The interlace mode of the image.   
npals | The number of palettes associated to the image.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMget_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad1d25d015457bfa2f00faf0952cc1f75)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gac06e374ebd82535e70fd14d214dc3f32)h5imget_npalettes_f()
subroutine h5imget_npalettes_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _image_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _npals_ ,   
|  | integer |  _errcode_ )  
Gets the number of palettes associated to an image.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
image_name | The name of the image dataset.   
npals | The number of palettes.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMget_npalettes()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga8644c36dbe768ea02d1ced695ab6b6b0)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga2069e82b8d26612f7e261ec90f7b45a2)h5imget_palette_f()
subroutine h5imget_palette_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _image_name_ ,   
|  | integer, intent(in) |  _pal_number_ ,   
|  | integer, dimension(*), intent(inout) |  _pal_data_ ,   
|  | integer |  _errcode_ )  
Gets the palette dataset.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
image_name | The name of the image dataset.   
pal_number | The zero based index that identifies the palette.   
pal_data | The palette dataset.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMget_palette_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga84c037ebf29227878e876740156fe666)h5imget_palette_info_f()
subroutine h5imget_palette_info_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _image_name_ ,   
|  | integer, intent(in) |  _pal_number_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), dimension(*), intent(inout) |  _pal_dims_ ,   
|  | integer |  _errcode_ )  
Gets information about a palette dataset (dimensions).  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
image_name | The name of the image dataset.   
pal_number | The zero based index that identifies the palette.   
pal_dims | The dimensions of the palette dataset.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMget_palette_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga96ac5556029f5723899d3ed55e7cc419)h5imis_image_f()
integer function h5imis_image_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ )  
Inquires if a dataset is an image.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset.  
See C API: [H5IMis_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gadd8c46c44a11fa36116a0b7314f248a8)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga9ac7575820d1b75c6661a78107f2eec5)h5imis_palette_f()
integer function h5imis_palette_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ )  
Inquires if a dataset is a palette. Returns zero (false), a positive (true) or a negative (failure) value.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset.  
See C API: [H5IMis_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga01d20431a40121ca9be11b6cb4b629a8)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga0986868cd2ddef408ac55d704b064a60)h5imlink_palette_f()
subroutine h5imlink_palette_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _image_name_ ,   
|  | character(len=*), intent(in) |  _pal_name_ ,   
|  | integer |  _errcode_ )  
This function attaches a palette to an existing image dataset.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
image_name | The name of the dataset to attach the palette to.   
pal_name | The name of the palette.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaa653878da63058bc7435f5ae7da8dc94)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gae17a8216ab5725dc684ed7baea996680)h5immake_image_24bit_f()
subroutine h5immake_image_24bit_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _width_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _height_ ,   
|  | character(len=*), intent(in) |  _il_ ,   
|  | integer, dimension(*), intent(in) |  _buf_ ,   
|  | integer |  _errcode_ )  
Creates and writes an image a 24 bit image.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to create.   
width | The width of the image.   
height | The height of the image.   
il | String defining the interlace mode.   
buf | Buffer with data to be written to the dataset.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMmake_image_24bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga151b78eea0bff9872af78f950c93c862)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gac30cbab36df84de283e30d9de9bdc848)h5immake_image_8bit_f()
subroutine h5immake_image_8bit_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _width_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _height_ ,   
|  | integer, dimension(*), intent(in) |  _buf_ ,   
|  | integer |  _errcode_ )  
Creates and writes an image an 8 bit image.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to create.   
width | The width of the image.   
height | The height of the image   
buf | Buffer with data to be written to the dataset   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMmake_image_8bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaed2cb8f411b58e6e48d596bba1d3b59b)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga8c66f1723f02ab514631b1ca448af532)h5immake_palette_f()
subroutine h5immake_palette_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _pal_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), dimension(*), intent(in) |  _pal_dims_ ,   
|  | integer, dimension(*), intent(in) |  _pal_data_ ,   
|  | integer |  _errcode_ )  
Creates and writes a palette.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
pal_name | The name of the palette.   
pal_dims | An array of the size of the palette dimensions.   
pal_data | Buffer with data to be written to the dataset.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMmake_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad6811d80238b25851cbde0475537b7d3)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#gace5754228364bcde3e21d6d1cdb0cb5e)h5imread_image_f()
subroutine h5imread_image_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer, dimension(*), intent(inout) |  _buf_ ,   
|  | integer |  _errcode_ )  
Reads image data from disk.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to create.   
buf | Buffer with data to store the image.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMread_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaec44462e614bb5b93d5776ca27491488)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html#ga6dfdaf798d1f36076aea143c1f23f8c2)h5imunlink_palette_f()
subroutine h5imunlink_palette_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _image_name_ ,   
|  | character(len=*), intent(in) |  _pal_name_ ,   
|  | integer |  _errcode_ )  
This function detaches a palette to an existing image dataset.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
image_name | The name of the image dataset.   
pal_name | The name of the palette.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5IMunlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga03f849a73219b982d8dbae2f1cbb7b7b)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


