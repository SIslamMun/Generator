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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title2 "H2")
  * [◆ H5IMget_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title3 "H2")
  * [◆ H5IMget_npalettes()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title4 "H2")
  * [◆ H5IMget_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title5 "H2")
  * [◆ H5IMget_palette_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title6 "H2")
  * [◆ H5IMis_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title7 "H2")
  * [◆ H5IMis_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title8 "H2")
  * [◆ H5IMlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title9 "H2")
  * [◆ H5IMmake_image_24bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title10 "H2")
  * [◆ H5IMmake_image_8bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title11 "H2")
  * [◆ H5IMmake_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title12 "H2")
  * [◆ H5IMread_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title13 "H2")
  * [◆ H5IMunlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#func-members)
HDF5 Images API (H5IM)
## Detailed Description
_Creating and manipulating HDF5 datasets intended to be interpreted as images (H5IM)_
The specification for the Images API is presented in another document: [HDF5 Image and Palette Specification Version 1.2](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html) This version of the API is primarily concerned with two dimensional raster data similar to HDF4 Raster Images. The HDF5 Images API uses the [HDF5 Lite APIs (H5LT,H5LD)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html). 

Note
     **Programming hints:**      To use any of these functions or subroutines, you must first include the relevant include file (C) or module (Fortran) in your application.       The following line includes the HDF5 Images package, H5IM, in C applications: 
#include "hdf5_hl.h"      This line includes the H5IM module in Fortran applications: 
use [h5im](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5im.html)
[h5im](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5im.html)
This module contains Fortran interfaces for H5IM.
**Definition** H5IMff.F90:35
  * [H5IMget_image_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad1d25d015457bfa2f00faf0952cc1f75)   
Gets information about an image dataset (dimensions, interlace mode and number of associated palettes).
  * [H5IMget_npalettes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga8644c36dbe768ea02d1ced695ab6b6b0)   
Gets the number of palettes associated to an image.
  * [H5IMget_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga89882863675a88899c327026237aecef)   
Gets the palette dataset.
  * [H5IMget_palette_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc)   
Gets information about a palette dataset (dimensions).
  * [H5IMis_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gadd8c46c44a11fa36116a0b7314f248a8)   
Inquires if a dataset is an image
  * [H5IMis_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga01d20431a40121ca9be11b6cb4b629a8)   
Inquires if a dataset is a palette.
  * [H5IMlink_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaa653878da63058bc7435f5ae7da8dc94)   
Attaches a palette to an image.
  * [H5IMmake_image_8bit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaed2cb8f411b58e6e48d596bba1d3b59b)   
Creates and writes an image.
  * [H5IMmake_image_24bit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga151b78eea0bff9872af78f950c93c862)   
Creates and writes a true color image.
  * [H5IMmake_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad6811d80238b25851cbde0475537b7d3)   
Creates and writes a palette.
  * [H5IMread_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaec44462e614bb5b93d5776ca27491488)   
Reads image data from disk.
  * [H5IMunlink_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga03f849a73219b982d8dbae2f1cbb7b7b)   
Detaches a palette from an image. 


##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMget_image_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad1d25d015457bfa2f00faf0952cc1f75) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *width, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *height, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *planes, char *interlace, [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) *npals)  
| Gets information about an image dataset (dimensions, interlace mode and number of associated palettes).   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMget_npalettes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga8644c36dbe768ea02d1ced695ab6b6b0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *image_name, [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) *npals)  
| Gets the number of palettes associated to an image.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMget_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga89882863675a88899c327026237aecef) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *image_name, int pal_number, unsigned char *pal_data)  
| Gets the palette dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMget_palette_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *image_name, int pal_number, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *pal_dims)  
| Gets information about a palette dataset (dimensions).   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMis_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gadd8c46c44a11fa36116a0b7314f248a8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name)  
| Inquires if a dataset is an image.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMis_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga01d20431a40121ca9be11b6cb4b629a8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name)  
| Inquires if a dataset is a palette.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMlink_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaa653878da63058bc7435f5ae7da8dc94) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *image_name, const char *pal_name)  
| Attaches a palette to an image.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMmake_image_24bit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga151b78eea0bff9872af78f950c93c862) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) width, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) height, const char *interlace, const unsigned char *buffer)  
| Creates and writes a true color image.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMmake_image_8bit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaed2cb8f411b58e6e48d596bba1d3b59b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) width, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) height, const unsigned char *buffer)  
| Creates and writes an image.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMmake_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad6811d80238b25851cbde0475537b7d3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *pal_name, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *pal_dims, const unsigned char *pal_data)  
| Creates and writes a palette.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMread_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaec44462e614bb5b93d5776ca27491488) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, unsigned char *buffer)  
| Reads image data from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5IMunlink_palette](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga03f849a73219b982d8dbae2f1cbb7b7b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *image_name, const char *pal_name)  
| Detaches a palette from an image.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad1d25d015457bfa2f00faf0952cc1f75)H5IMget_image_info()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMget_image_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _width_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _height_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _planes_ ,   
|  | char * |  _interlace_ ,   
|  |  [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) * |  _npals_ )  
Gets information about an image dataset (dimensions, interlace mode and number of associated palettes). 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset   
[out] | width | The width of the image   
[out] | height | The height of the image   
[out] | planes | The number of color planes of the image   
[out] | interlace | The interlace mode of the image   
[out] | npals | The number of palettes associated to the image 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMget_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad1d25d015457bfa2f00faf0952cc1f75 "Gets information about an image dataset \(dimensions, interlace mode and number of associated palettes...") gets information about an image named `dset_name` attached to the file or group specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga8644c36dbe768ea02d1ced695ab6b6b0)H5IMget_npalettes()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMget_npalettes  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _image_name_ ,   
|  |  [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) * |  _npals_ )  
Gets the number of palettes associated to an image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | image_name | The name of the image dataset   
[out] | npals | The number of palettes 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMget_npalettes()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga8644c36dbe768ea02d1ced695ab6b6b0 "Gets the number of palettes associated to an image.") gets the number of palettes associated to an image specified by `image_name`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga89882863675a88899c327026237aecef)H5IMget_palette()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMget_palette  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _image_name_ ,   
|  | int |  _pal_number_ ,   
|  | unsigned char * |  _pal_data_ )  
Gets the palette dataset. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | image_name | The name of the image dataset   
[in] | pal_number | The zero based index that identifies the palette   
[out] | pal_data | The palette dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMget_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga89882863675a88899c327026237aecef "Gets the palette dataset.") gets the palette dataset identified by `pal_number` (a zero based index) associated to an image specified by `image_name`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc)H5IMget_palette_info()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMget_palette_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _image_name_ ,   
|  | int |  _pal_number_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _pal_dims_ )  
Gets information about a palette dataset (dimensions). 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | image_name | The name of the image dataset   
[in] | pal_number | The zero based index that identifies the palette   
[out] | pal_dims | The dimensions of the palette dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMget_palette_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga2c669d6a99cd1f3ea0d5005cd106f5dc "Gets information about a palette dataset \(dimensions\).") gets the dimensions of the palette dataset identified by `pal_number` (a zero based index) associated to an image specified by `image_name`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gadd8c46c44a11fa36116a0b7314f248a8)H5IMis_image()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMis_image  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ )  
Inquires if a dataset is an image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5IMis_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gadd8c46c44a11fa36116a0b7314f248a8 "Inquires if a dataset is an image.") inquires if a dataset named `dset_name`, attached to the file or group specified by the identifier `loc_id`, is an image based on the HDF5 Image and Palette Specification. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga01d20431a40121ca9be11b6cb4b629a8)H5IMis_palette()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMis_palette  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ )  
Inquires if a dataset is a palette. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5IMis_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga01d20431a40121ca9be11b6cb4b629a8 "Inquires if a dataset is a palette.") inquires if a dataset named `dset_name`, attached to the file or group specified by the identifier `loc_id`, is a palette based on the HDF5 Image and Palette Specification. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaa653878da63058bc7435f5ae7da8dc94)H5IMlink_palette()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMlink_palette  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _image_name_ ,   
|  | const char * |  _pal_name_ )  
Attaches a palette to an image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | image_name | The name of the dataset to attach the palette to   
[in] | pal_name | The name of the palette 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaa653878da63058bc7435f5ae7da8dc94 "Attaches a palette to an image.") attaches a palette named `pal_name` to an image specified by `image_name`. The image dataset may or not already have an attached palette. If it has, the array of palette references is extended to hold the reference to the new palette. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga151b78eea0bff9872af78f950c93c862)H5IMmake_image_24bit()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMmake_image_24bit  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _width_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _height_ ,   
|  | const char * |  _interlace_ ,   
|  | const unsigned char * |  _buffer_ )  
Creates and writes a true color image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to create   
[in] | width | The width of the image   
[in] | height | The height of the image   
[in] | interlace | String defining the interlace mode   
[in] | buffer | Buffer with data to be written to the dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMmake_image_24bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga151b78eea0bff9872af78f950c93c862 "Creates and writes a true color image.") creates and writes a dataset named `dset_name` attached to the file or group specified by the identifier `loc_id`. This function defines a true color image conforming to the HDF5 Image and Palette specification. The function assumes that the image data is of the type [H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd).
A true color image is an image where the pixel storage contains several color planes. In a 24 bit RGB color model, these planes are red, green and blue. In a true color image the stream of bytes can be stored in several different ways, thus defining the interlace (or interleaving) mode. The 2 most used types of interlace mode are interlace by pixel and interlace by plane. In the 24 bit RGB color model example, interlace by plane means all the red components for the entire dataset are stored first, followed by all the green components, and then by all the blue components. Interlace by pixel in this example means that for each pixel the sequence red, green, blue is defined. In this function, the interlace mode is defined in the parameter `interlace`, a string that can have the values INTERLACE_PIXEL or INTERLACE_PLANE. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaed2cb8f411b58e6e48d596bba1d3b59b)H5IMmake_image_8bit()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMmake_image_8bit  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _width_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _height_ ,   
|  | const unsigned char * |  _buffer_ )  
Creates and writes an image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to create   
[in] | width | The width of the image   
[in] | height | The height of the image   
[in] | buffer | Buffer with data to be written to the dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMmake_image_8bit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaed2cb8f411b58e6e48d596bba1d3b59b "Creates and writes an image.") creates and writes a dataset named `dset_name` attached to the file or group specified by the identifier `loc_id`. Attributes conforming to the HDF5 Image and Palette specification for an indexed image are attached to the dataset, thus identifying it as an image. The image data is of the type [H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd). An indexed image is an image in which each each pixel information storage is an index to a table palette. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad6811d80238b25851cbde0475537b7d3)H5IMmake_palette()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMmake_palette  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _pal_name_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _pal_dims_ ,   
|  | const unsigned char * |  _pal_data_ )  
Creates and writes a palette. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | pal_name | The name of the palette   
[in] | pal_dims | An array of the size of the palette dimensions   
[in] | pal_data | Buffer with data to be written to the dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMmake_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gad6811d80238b25851cbde0475537b7d3 "Creates and writes a palette.") creates and writes a dataset named `pal_name`. Attributes conforming to the HDF5 Image and Palette specification are attached to the dataset, thus identifying it as a palette. The palette data is of the type [H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaec44462e614bb5b93d5776ca27491488)H5IMread_image()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMread_image  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | unsigned char * |  _buffer_ )  
Reads image data from disk. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to create   
[out] | buffer | Buffer with data to store the image 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMread_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#gaec44462e614bb5b93d5776ca27491488 "Reads image data from disk.") reads a dataset named `dset_name` attached to the file or group specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga03f849a73219b982d8dbae2f1cbb7b7b)H5IMunlink_palette()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5IMunlink_palette  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _image_name_ ,   
|  | const char * |  _pal_name_ )  
Detaches a palette from an image. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | image_name | The name of the image dataset   
[in] | pal_name | The name of the palette 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5IMunlink_palette()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html#ga03f849a73219b982d8dbae2f1cbb7b7b "Detaches a palette from an image.") detaches a palette from an image specified by `image_name`. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


