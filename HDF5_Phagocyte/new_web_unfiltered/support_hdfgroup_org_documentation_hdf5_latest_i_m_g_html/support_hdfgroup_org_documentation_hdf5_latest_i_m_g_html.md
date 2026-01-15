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
  * [ HDF5 Image Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title0 "H1")
  * [ Overview](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title1 "H2")
  * [ Image Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title2 "H2")
  * [ Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title3 "H2")
  * [ Storage Layout and Properties for Images](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title4 "H2")
  * [ HDF5 Palette Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title5 "H1")
  * [ Overview](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title6 "H2")
  * [ Palette Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title7 "H2")
  * [ Storage Layout for Palettes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title8 "H2")
  * [ Consistency and Correlation of Image and Palette Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#title9 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Image and Palette Specification Version 1.2
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
* * *
The HDF5 specification defines the standard objects and storage for the standard HDF5 objects. (For information about the HDF5 library, model and specification, see the HDF documentation.) This document is an additional specification to define a standard profile for how to store image data in HDF5. Image data in HDF5 is stored as HDF5 datasets with standard attributes to define the properties of the image.
This specification is primarily concerned with two dimensional raster data similar to HDF4 Raster Images. Specifications for storing other types of imagery will be covered in other documents. This specification defines: /li Standard storage and attributes for an Image dataset ([Overview](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_image_spec_spec_over)) /li Standard storage and attributes for Palettes ([Image Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_image_spec_spec_attr)) /li Standard for associating Palettes with Images. ([Consistency and Correlation of Image and Palette Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#sec_tab_spec_sect3))
#  HDF5 Image Specification
##  Overview
Image data is stored as an HDF5 dataset with values of HDF5 class Integer or Float. A common example would be a two dimensional dataset, with elements of class Integer, e.g., a two dimensional array of unsigned 8 bit integers. However, this specification does not limit the dimensions or number type that may be used for an Image.
The dataset for an image is distinguished from other datasets by giving it an attribute "CLASS=IMAGE". In addition, the Image dataset may have an optional attribute "PALETTE" that is an array of object references for zero or more palettes. The Image dataset may have additional attributes to describe the image data, as defined in [Image Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_image_spec_spec_attr).
A Palette is an HDF5 dataset which contains color map information. A Palette dataset has an attribute "CLASS=PALETTE" and other attributes indicating the type and size of the palette, as defined in [HDF5 Palette Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#sec_tab_spec_sect2). A Palette is an independent object, which can be shared among several Image datasets.
##  Image Attributes
The attributes for the Image are scalars unless otherwise noted. The length of String valued attributes should be at least the number of characters. Optionally, String valued attributes may be stored in a String longer than the minimum, in which case it must be zero terminated or null padded. "Required" attributes must always be used. "Optional" attributes must be used when required.
##  Attributes
**Table 1. Attributes of an Image Dataset** **Attribute Name** |  **Required** or **Optional** |  **Type** |  **String Size** |  **Value** | Description   
---|---|---|---|---|---  
CLASS  | Required  | String  | 5  | "IMAGE"  | This attribute is type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size 5. For all Images, the value of this attribute is "IMAGE". This attribute identifies this data set as intended to be interpreted as an image that conforms to the specifications on this page.   
PALETTE  | Optional  | Array Object References  |  | <references to Palette datasets>1 | A Image dataset within an HDF5 file may optionally specify an array of palettes to be viewed with. The dataset will have an attribute field called "<b>PALETTE</b>" which contains a one-dimensional array of object reference pointers (HDF5 datatype [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1)) which refer to palettes in the file. The palette datasets must conform to the Palette specification in [HDF5 Palette Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#sec_tab_spec_sect2). The first palette in this array will be the default palette that the data may be viewed with.   
IMAGE_SUBCLASS  | Optional2 | String  | 15,  
12,  
15,  
13  | "IMAGE_GRAYSCALE",  
"IMAGE_BITMAP",  
"IMAGE_TRUECOLOR",  
"IMAGE_INDEXED"  | If present, the value of this attribute indicates the type of Palette that should be used with the Image. This attribute is a scalar of type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size according to the string plus one. The values are:  

  * IMAGE_GRAYSCALE (length 15) A grayscale image 
  * IMAGE_BITMAP (length 12) A bit map image 
  * IMAGE_TRUECOLOR (length 15) A truecolor image 
  * IMAGE_INDEXED (length 13) An indexed image

  
INTERLACE_MODE  | Optional3,6 | String  | 15  | The layout of components if more than one component per pixel.  | For images with more than one component for each pixel, this optional attribute specifies the layout of the data. The values are type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) of length
  1. See [Storage Layout and Properties for Images](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_image_spec_spec_store) for information about the storage layout for data.  

     * INTERLACE_PIXEL (default): the component values for a pixel are contiguous. 
     * INTERLACE_PLANE: each component is stored as a plane.

  
DISPLAY_ORIGIN  | Optional  | String  | 2  | If set, indicates the intended location of the pixel (0,0).  | This optional attribute indicates the intended orientation of the data on a two-dimensional raster display. The value indicates which corner the pixel at (0, 0) should be viewed. The values are type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) of length 2. If DISPLAY_ORIGIN is not set, the orientation is undefined.  

  * UL: (0,0) is at the upper left. 
  * LL: (0,0) is at the lower left. 
  * UR: (0,0) is at the upper right. 
  * LR: (0,0) is at the lower right.

  
IMAGE_WHITE_IS_ZERO  | Optional3,4 | Unsigned Integer  |  | 0 = false, 1 = true  | This attribute is of type [H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd). 0 = false, 1 = true . This is used for images with IMAGE_SUBCLASS="IMAGE_GRAYSCALE" or "IMAGE_BITMAP".   
IMAGE_MINMAXRANGE  | Optional3,5 | Array [2] <same datatype as data values> |  | The (<minimum>, <maximum>) value of the data.  | If present, this attribute is an array of two numbers, of the same HDF5 datatype as the data. The first element is the minimum value of the data, and the second is the maximum. This is used for images with IMAGE_SUBCLASS="IMAGE_GRAYSCALE", "IMAGE_BITMAP" or "IMAGE_INDEXED".   
IMAGE_BACKGROUNDINDEX  | Optional3 | Unsigned Integer  |  | The index of the background color.  | If set, this attribute indicates the index value that should be interpreted as the "background color". This attribute is HDF5 type [H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5).   
IMAGE_TRANSPARENCY  | Optional3,5 | Unsigned Integer  |  | The index of the transparent color.  | If set, this attribute indicates the index value that should be interpreted as the "transparent color". This attribute is HDF5 type [H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5). This attribute may not be used for IMAGE_SUBCLASS="IMAGE_TRUECOLOR".   
IMAGE_ASPECTRATIO  | Optional3,4 | Unsigned Integer  |  | The aspect ratio.  | If set, this attribute indicates the aspect ratio.   
IMAGE_COLORMODEL  | Optional3,6 | String  | 3, 4, or 5  | The color model, as defined below in the Palette specification for attribute **PAL_COLORMODEL**.  | If set, this attribute indicates the color model of Palette that should be used with the Image. This attribute is of type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size 3, 4, or 5. The value is one of the color models described in the Palette specification in [Palette Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_tab_spec_sect2_22). This attribute may be used only for IMAGE_SUBCLASS="IMAGE_TRUECOLOR" or "IMAGE_INDEXED".   
IMAGE_GAMMACORRECTION  | Optional3,6 | Float  |  | The gamma correction.  | If set, this attribute gives the Gamma correction. The attribute is type [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c). This attribute may be used only for IMAGE_SUBCLASS="IMAGE_TRUECOLOR" or "IMAGE_INDEXED".   
IMAGE_VERSION  | Required  | String  | 3  | "1.2"  | This attribute is of type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size corresponding to the length of the version string. This attribute identifies the version number of this specification to which it conforms. The current version number is "1.2".   
#### Notes
  * 1. The first element of the array is the defaultPalette. 
  * 2. This attribute is **required** for images that use one of the standard color map types listed. 
  * 3. This attribute is **required** if set for the source image, in the case that the image is translated from another file into HDF5. 
  * 4. This applies to: IMAGE_SUBCLASS="IMAGE_GRAYSCALE" or "IMAGE_BITMAP". 
  * 5. This applies to: IMAGE_SUBCLASS="IMAGE_GRAYSCALE", "IMAGE_BITMAP", or "IMAGE_INDEXED". 
  * 6. This applies to: IMAGE_SUBCLASS="IMAGE_TRUECOLOR", or "IMAGE_INDEXED".


Table 2 summarizes the standard attributes for an Image dataset(s) using the common sub-classes. Required means that the attribute listed on the leftmost column is required for the image subclass on the first row, Optional means that the attribute is optional for that subclass and NA that the attribute cannot be applied to that subclass. The two first rows show the only required attributes for all subclasses.   

**Table 2a. Applicability of Attributes to IMAGE sub-classes** **IMAGE_SUBCLASS** 1 |  **IMAGE_GRAYSCALE** |  **IMAGE_BITMAP**  
---|---|---  
CLASS  | Required  | Required   
IMAGE_VERSION  | Required  | Required   
INTERLACE_MODE  | NA  | NA   
IMAGE_WHITE_IS_ZERO  | Required  | Required   
IMAGE_MINMAXRANGE  | Optional  | Optional   
IMAGE_BACKGROUNDINDEX  | Optional  | Optional   
IMAGE_TRANSPARENCY  | Optional  | Optional   
IMAGE_ASPECTRATIO  | Optional  | Optional   
IMAGE_COLORMODEL  | NA  | NA   
IMAGE_GAMMACORRECTION  | NA  | NA   
PALETTE  | Optional  | Optional   
DISPLAY_ORIGIN  | Optional  | Optional   
**Table 2b. Applicability of Attributes to IMAGE sub-classes** **IMAGE_SUBCLASS** |  **IMAGE_TRUECOLOR** |  **IMAGE_INDEXED**  
---|---|---  
CLASS  | Required  | Required   
IMAGE_VERSION  | Required  | Required   
INTERLACE_MODE  | Required  | NA   
IMAGE_WHITE_IS_ZERO  | NA  | NA   
IMAGE_MINMAXRANGE  | NA  | Optional   
IMAGE_BACKGROUNDINDEX  | NA  | Optional   
IMAGE_TRANSPARENCY  | NA  | Optional   
IMAGE_ASPECTRATIO  | Optional  | Optional   
IMAGE_COLORMODEL  | Optional  | Optional   
IMAGE_GAMMACORRECTION  | Optional  | Optional   
PALETTE  | Optional  | Optional   
DISPLAY_ORIGIN  | Optional  | Optional   
##  Storage Layout and Properties for Images
In the case of an image with more than one component per pixel (e.g., Red, Green, and Blue), the data may be arranged in one of two ways. Following HDF4 terminology, the data may be interlaced by pixel or by plane, which should be indicated by the INTERLACE_MODE attribute. In both cases, the dataset will have a dataspace with three dimensions, height, width, and components. The interlace modes specify different orders for the dimensions. 
**Table 3. Storage of multiple component image data.** **Interlace Mode** |  **Dimensions in the Dataspace**  
---|---  
INTERLACE_PIXEL  | [height][width][pixel components]   
INTERLACE_PLANE  | [pixel components][height][width]   
For example, consider a 5 (rows) by 10 (column) image, with Red, Green, and Blue components. Each component is an unsigned byte. In HDF5, the datatype would be declared as an unsigned 8 bit integer. For pixel interlace, the dataspace would be a three dimensional array, with dimensions: [10][5][3]. For plane interleave, the dataspace would be three dimensions: [3][10][5].
In the case of images with only one component, the dataspace may be either a two dimensional array, or a three dimensional array with the third dimension of size 1. For example, a 5 by 10 image with 8 bit color indexes would be an HDF5 dataset with type unsigned 8 bit integer. The dataspace could be either a two dimensional array, with dimensions [10][5], or three dimensions, with dimensions either [10][5][1] or [1][10][5]. 
Image datasets may be stored with any chunking or compression properties supported by HDF5.
**A note concerning compatibility with HDF5 GR interface:** An Image dataset is stored as an HDF5 dataset. It is important to note that the order of the dimensions is the same as for any other HDF5 dataset. For a two dimensional image that is to be stored as a series of horizontal scan lines, with the scan lines contiguous (i.e., the fastest changing dimension is 'width'), the image will have a dataspace with _dim[0] = height_ and _dim[1]_ = _width_. This is completely consistent with all other HDF5 datasets.
Users familiar with HDF4 should be cautioned that _this is not the same as HDF4_ , and specifically is not consistent with what the HDF4 GR interface does.
#  HDF5 Palette Specification
##  Overview
A palette is the means by which color is applied to an image and is also referred to as a color lookup table. It is a table in which every row contains the numerical representation of a particular color. In the example of an 8 bit standard RGB color model palette, this numerical representation of a color is presented as a triplet specifying the intensity of red, green, and blue components that make up each color. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/Palettes_fm_anc.gif)  
---  
In this example, the color component numeric type is an 8 bit unsigned integer. While this is most common and recommended for general use, other component color numeric datatypes, such as a 16 bit unsigned integer , may be used. This type is specified as the type attribute of the palette dataset.
The minimum and maximum values of the component color numeric are specified as attribute of the palette dataset. See below (attribute PAL_MINMAXNUMERIC). If these attributes do not exist, it is assumed that the range of values will fill the space of the color numeric type. i.e. with an 8 bit unsigned integer, the valid range would be 0 to 255 for each color component.
The HDF5 palette specification additionally allows for color models beyond RGB. YUV, HSV, CMY, CMYK, YCbCr color models are supported, and may be specified as a color model attribute of the palette dataset. _(see[Palette Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_i_m_g.html#subsec_tab_spec_sect2_22) for details)_.
In HDF 4 and earlier, palettes were limited to 256 colors. The HDF5 palette specification allows for palettes of varying length. The length is specified as the number of rows of the palette dataset. 
Important Note: The specification of the Indexed Palette will change substantially in the next version. The Palette described here is _deprecated_ and is not supported.   
---  
_DEPRECATED_  
In a standard palette, the color entries are indexed directly. HDF5 supports the notion of a range index table. Such a table defines an ascending ordered list of ranges that map dataset values to the palette. If a range index table exists for the palette, the PAL_TYPE attribute will be set to "RANGEINDEX", and the PAL_RANGEINDEX attribute will contain an object reference to a range index table array. If not, the PAL_TYPE attribute either does not exist, or will be set to "STANDARD". The range index table array consists of a one dimensional array with the same length as the palette dataset - 1. Ideally, the range index would be of the same type as the dataset it refers to, however this is not a requirement. **Example 2: A range index array of type floating point** ![](https://support.hdfgroup.org/documentation/hdf5/latest/PaletteExample1.gif) The range index array attribute defines the "<i>to</i>" of the range. Notice that the range index array attribute is one less entry in size than the palette. The first entry of 0.1259, specifies that all values below and up to 0.1259 inclusive, will map to the first palette entry. The second entry signifies that all values greater than 0.1259 up to 0.3278 inclusive, will map to the second palette entry, etc. All value greater than the last range index array attribute (100000) map to the last entry in the palette.   
##  Palette Attributes
A palette exists in an HDF file as an independent data set with accompanying attributes. The Palette attributes are scalars except where noted otherwise. String values should have size the length of the string value plus one. "Required" attributes must be used. "Optional" attributes must be used when required.
These attributes are defined as follows: 
**Table 4. Attributes of a Palette Dataset** **Attribute Name** |  **Required or Optional** |  **Type** |  **String Size** |  **Value** | Description   
---|---|---|---|---|---  
CLASS  | Required  | String  | 7  | "PALETTE"  | This attribute is of type H5T_C_S1, with size 7.For all palettes, the value of this attribute is "PALETTE". This attribute identifies this palette data set as a palette that conforms to the specifications on this page.   
PAL_COLORMODEL  | Required  | String  | 3, 4, or 5  | Color Model: "RGB", "YUV", "CMY", "CMYK", "YCbCr", or "HSV"  | This attribute is of type H5T_C_S1, with size 3, 4, or 5. Possible values for this are "RGB", "YUV", "CMY", "CMYK", "YCbCr", "HSV".  
This defines the color model that the entries in the palette data set represent.  

"RGB" 
     Each color index contains a triplet where the first value defines the red component, second defines the green component, and the third the blue component. 

"CMY" 
     Each color index contains a triplet where the first value defines the cyan component, second defines the magenta component, and the third the yellow component. 

"CMYK" 
     Each color index contains a quadruplet where the first value defines the cyan component, second defines the magenta component, the third the yellow component, and the forth the black component. 

"YCbCr" 
     Class Y encoding model. Each color index contains a triplet where the first value defines the luminance, second defines the Cb Chromonance, and the third the Cr Chromonance. 

"YUV" 
     Composite encoding color model. Each color index contains a triplet where the first value defines the luminance component, second defines the chromonance component, and the third the value component. 

"HSV" 
    Each color index contains a triplet where the first value defines the hue component, second defines the saturation component, and the third the value component. The hue component defines the hue spectrum with a low value representing magenta/red progressing to a high value which would represent blue/magenta, passing through yellow, green, cyan. A low value for the saturation component means less color saturation than a high value. A low value for _value_ will be darker than a high value.   
PAL_TYPE  | Required  | String  | 9  
| or 10  
---  
"STANDARD8"  
| or "RANGEINDEX" _(Deprecated)_  
---  
This attribute is of type H5T_C_S1, with size 9 or 10. The current supported values for this attribute are: "STANDARD8" or "RANGEINDEX".  
A PAL_TYPE of "STANDARD8" defines a palette dataset such that the first entry defines index 0, the second entry defines index 1, etc. up until the length of the palette - 1. This assumes an image dataset with direct indexes into the palette.  
|  _Deprecated_ If the PAL_TYPE is set to "RANGEINDEX", there will be an additional attribute with a name of **PAL_RANGEINDEX** , (See example 2 for more details)   
---  
  


Attribute name="<b>PAL_RANGEINDEX</b>" _(Deprecated)_ 
    The **PAL_RANGEINDEX** attribute contains an HDF object reference (HDF5 datatype H5T_STD_REF_OBJ) pointer which specifies a range index array in the file to be used for color lookups for the palette. (Only for PAL_TYPE="RANGEINDEX")   
---  
|  _Deprecated_  
RANGE_INDEX  
---  
| _Deprecated_  
---  
| Object Reference  
---  
| _Deprecated_  
---  
| <Object Reference to Dataset of range index values>  
---  
| _Deprecated_  
---  
PAL_MINMAXNUMERIC  | Optional  | Array[2] of <same datatype as palette> |  | The first value is the <Minimum value for color values>, the second value is <Maximum value for color values>2 | If present, this attribute is an array of two numbers, of the same HDF5 datatype as the palette elements or color numerics. They specify the minimum and maximum values of the color numeric components. For example, if the palette was an RGB of type Float, the color numeric range for Red, Green, and Blue could be set to be between 0.0 and 1.0. The intensity of the color guns would then be scaled accordingly to be between this minimum and maximum attribute.   
PAL_VERSION  | Required  | String  | 4  | "1.2"  | This attribute is of type H5T_C_S1, with size corresponding to the length of the version string. This attribute identifies the version number of this specification to which it conforms. The current version is "1.2".   
#### Notes
  * 1. The RANGE_INDEX attribute is required if the PAL_TYPE is "RANGEINDEX". Otherwise, the RANGE_INDEX attribute should be omitted. (Range index is deprecated.) 
  * 2. The minimum and maximum are optional. If not set, the range is assumed to the maximum range of the number type. If one of these attributes is set, then both should be set. The value of the minimum must be less than or equal to the value of the maximum.


Table 5 summarized the uses of the standard attributes for a palette dataset. Required means that the attribute listed on the leftmost column is required for the palette type on the first row, Optional means that the attribute is optional for that type and NA that the attribute cannot be applied to that type. The four first rows show the attributes that are always required for the two palette types.
**Table 5. Applicability of Attributes** **PAL_TYPE** |  **STANDARD8** |  **RANGEINDEX**  
---|---|---  
CLASS  | Required  | Required   
PAL_VERSION  | Required  | Required   
PAL_COLORMODEL  | Required  | Required   
RANGE_INDEX  | NA  | Required   
PAL_MINMAXNUMERIC  | Optional  | Optional   
##  Storage Layout for Palettes
The values of the Palette are stored as a dataset. The datatype can be any HDF5 atomic numeric type. The dataset will have dimensions (`nentries` by `ncomponents`), where '`nentries`' is the number of colors (usually 256) and '`ncomponents'` is the number of values per color (3 for **RGB** , 4 for **CMYK** , etc.)
#  Consistency and Correlation of Image and Palette Attributes
The objects in this specification are an extension to the base HDF5 specification and library. They are accessible with the standard HDF5 library, but the semantics of the objects are not enforced by the base library. For example, it is perfectly possible to add an attribute called **IMAGE** to _any_ dataset, or to include an object reference to _any_ HDF5 dataset in a **PALETTE** attribute. This would be a valid HDF5 file, but not conformant to this specification. The rules defined in this specification must be implemented with appropriate software, and applications must use conforming software to assure correctness.
The Image and Palette specifications include several redundant standard attributes, such as the **IMAGE_COLORMODEL** and the **PAL_COLORMODEL**. These attributes are informative not normative, in that it is acceptable to attach a Palette to an Image dataset even if their attributes do not match. Software is not required to enforce consistency, and files may contain mismatched associations of Images and Palettes. In all cases, it is up to applications to determine what kinds of images and color models can be supported.
For example, an Image that was created from a file with an "RGB" may have a "YUV" Palette in its **PALETTE** attribute array. This would be a legal HDF5 file and also conforms to this specification, although it may or may not be correct for a given application.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


