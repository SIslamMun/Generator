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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title3 "H2")
  * [◆ H5Zfilter_avail()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title4 "H2")
  * [◆ H5Zget_filter_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title5 "H2")
  * [◆ H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title6 "H2")
  * [◆ H5Zunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#func-members)
Filters (H5Z)
## Detailed Description
Use the functions in this module to manage HDF5 filters.
User-defined filters are created by registering a filter descriptor of type [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) with the library.
Available filters can be read or examined at runtime.
It is conceivable that filters are stateful and that that state be updated at runtime.
Filters are deleted by unregistering.
Create | Read   
---|---  
10size_t 11filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t *buf_size, 12 void **buf) 13{ 14 buf_size = 0; 15 16 if (flags & [H5Z_FLAG_REVERSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a843859882318f69c9b4a38d3a88e0bfb)) { 17 // read data, e.g., decompress data 18 // ... 19 } 20 else { 21 // write data, e.g., compress data 22 // ... 23 } 24 25 return nbytes; 26} 35 { 36 __label__ fail_register; 37 [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) cls; 38 cls.version = [H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9); 39 cls.id = 256; 40 cls.encoder_present = 1; 41 cls.decoder_present = 1; 42 cls.name = "Identity filter"; 43 cls.can_apply = NULL; 44 cls.set_local = NULL; 45 cls.filter = &filter; 46 47 // register the filter 48 if ([H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b)(&cls) < 0) { 49 ret_val = EXIT_FAILURE; 50 goto fail_register; 51 } 52 53 // do something with filter 54 // ... 55 56 // unregister the filter if no longer required 57 // ... 58 59fail_register:; 60 } |  64 { 65 __label__ fail_avail; 66 67 [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) flt = [H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce); 68 unsigned flags = 0; 69 70 // check if the deflate filter is available 71 if ([H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee)(flt) < 0) { 72 ret_val = EXIT_FAILURE; 73 goto fail_avail; 74 } 75 // retrieve the deflate filter info 76 if ([H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba)(flt, &flags) < 0) { 77 ret_val = EXIT_FAILURE; 78 goto fail_avail; 79 } 80 81 // check if the deflate encoder or decoder is enabled 82 if ([H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a) & flags) 83 printf("Deflate encoder enabled.\n"); 84 if ([H5Z_FILTER_CONFIG_DECODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a4dead61ceb139a3f97505d6e52eb1b8a) & flags) 85 printf("Deflate decoder enabled.\n"); 86 87fail_avail:; 88 }  
Update | Delete   
92 { 93 // N/A |  97 { 98 // unregister the identity filter 99 if ([H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8)(256) < 0) { 100 ret_val = EXIT_FAILURE; 101 } 102 }  
HDF5 supports a filter pipeline that provides the capability for standard and customized raw data processing during I/O operations. HDF5 is distributed with a small set of standard filters such as compression (gzip, SZIP, and a shuffling algorithm) and error checking (Fletcher32 checksum). For further flexibility, the library allows a user application to extend the pipeline through the creation and registration of customized filters. See [HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#sec_filter_plugins)
The flexibility of the filter pipeline implementation enables the definition of additional filters by a user application. A filter 
  * is associated with a dataset when the dataset is created, 
  * can be used only with chunked data (i.e., datasets stored in the [H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262) storage layout), and 
  * is applied independently to each chunk of the dataset.


The HDF5 library does not support filters for contiguous datasets because of the difficulty of implementing random access for partial I/O. Compact dataset filters are not supported because they would not produce significant results.
HDF5 allows chunked data to pass through user-defined filters on the way to or from disk. The filters operate on chunks of an [H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262) dataset can be arranged in a pipeline so output of one filter becomes the input of the next filter.
Each filter has a two-byte identification number (type [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237 "Filter identifiers.")) allocated by The HDF Group and can also be passed application-defined integer resources to control its behavior. Each filter also has an optional ASCII comment string.
Values for [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237 "Filter identifiers.") | Description   
---|---  
`0-255` | These values are reserved for filters predefined and registered by the HDF5 library and of use to the general public.   
`256-511` | Filter values in this range are intended for testing only and can be temporarily used by any organization. No attempts are made to resolve numbering conflicts, as all definitions are temporary.   
`512-32,767` | Filter values within this range are designated for filters managed by The HDF Group, but they are nominally requested, developed, and supported by third parties. Please contact the [HDF5 development team](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html) to reserve a value or range of values for use by your filters.   
`32,768-65,535` | Filter values in this range are designated for internal company use or application testing when assessing a feature. The HDF Group does not track or document the use of filters within this range.   
Filter identifiers for the filters distributed with the HDF5 Library are as follows: ! [PreDefFilters] 
[H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce) | The gzip compression, or deflation, filter   
---|---  
[H5Z_FILTER_SZIP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a421d9941c68ebb776573baeb9aa77cd2) | The SZIP compressionfilter   
[H5Z_FILTER_NBIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a8cc463fa1979bd4bfa0dd9aa6a41e49d) | The N-bit compression filter   
[H5Z_FILTER_SCALEOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a745d2ccb4f7712ed78ef5e562e27d2ca) | The scale-offset compression filter   
[H5Z_FILTER_SHUFFLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#aa723f1a71601bf22c95620a490ecf1af) | The shuffle algorithm filter   
[H5Z_FILTER_FLETCHER32](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a59ca894c9c2b99b1614b0c46a7407f1c) | The Fletcher32 checksum, or error checking, filter   
! [PreDefFilters]
Custom filters that have been registered with the library will have additional unique identifiers.
See [HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#sec_filter_plugins) for more information on how an HDF5 application can apply a filter that is not registered with the HDF5 library. 
##  Topics  
---  
| [Predefined Filters](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z_p_r_e.html)  
##  Functions  
---  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee) ([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) id)  
| Determines whether a filter is available.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba) ([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) filter, unsigned int *filter_config_flags)  
| Retrieves information about a filter.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b) (const void *cls)  
| Registers a new filter with the HDF5 library.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8) ([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) id)  
| Unregisters a filter.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee)H5Zfilter_avail()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Zfilter_avail  | ( | [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) | _id_ | ) |   
---|---|---|---|---|---  
Determines whether a filter is available.  

Parameters
     [in] | id | Filter identifier   
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5Zfilter_avail()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee "Determines whether a filter is available.") determines whether the filter specified in `id` is available to the application. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba)H5Zget_filter_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Zget_filter_info  | ( | [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) |  _filter_ ,   
---|---|---|---  
|  | unsigned int * |  _filter_config_flags_ )  
Retrieves information about a filter.  

Parameters
     [in] | filter | Filter identifier   
---|---|---  
[out] | filter_config_flags | A bit field encoding the returned filter information  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Zget_filter_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba "Retrieves information about a filter.") retrieves information about a filter. At present, this means that the function retrieves a filter's configuration flags, indicating whether the filter is configured to decode data, encode data, neither, or both.
If `filter_config_flags` is not set to NULL prior to the function call, the returned parameter contains a bit field specifying the available filter configuration. The configuration flag values can then be determined through a series of bitwise AND operations, as described below.
Valid filter configuration flags include the following: 
[H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a) | Encoding is enabled for this filter   
---|---  
[H5Z_FILTER_CONFIG_DECODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a4dead61ceb139a3f97505d6e52eb1b8a) | Decoding is enabled for this filter   
A bitwise AND of the returned `filter_config_flags` and a valid filter configuration flag will reveal whether the related configuration option is available. For example, if the value of 
[H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a) & filter_config_flags
[H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a)
#define H5Z_FILTER_CONFIG_ENCODE_ENABLED
**Definition** H5Zpublic.h:270
is true, i.e., greater than 0 (zero), the queried filter is configured to encode data; if the value is `false`, i.e., equal to 0 (zero), the filter is not so configured.
If a filter is not encode-enabled, the corresponding `H5Pset_*` function will return an error if the filter is added to a dataset creation property list (which is required if the filter is to be used to encode that dataset). For example, if the [H5Z_FILTER_CONFIG_ENCODE_ENABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ac4ec01a86fdac6619c7c3c1fcf3bf86a) flag is not returned for the SZIP filter, [H5Z_FILTER_SZIP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a421d9941c68ebb776573baeb9aa77cd2), a call to [H5Pset_szip()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga37de4b6071a94574cfab5cd6de9c3fc6 "Sets up use of the SZIP compression filter.") will fail.
If a filter is not decode-enabled, the application will not be able to read an existing file encoded with that filter.
This function should be called, and the returned `filter_config_flags` should be analyzed, before calling any other function, such as [H5Pset_szip()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga37de4b6071a94574cfab5cd6de9c3fc6 "Sets up use of the SZIP compression filter."), that might require a particular filter configuration. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b)H5Zregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Zregister  | ( | const void * | _cls_ | ) |   
---|---|---|---|---|---  
Registers a new filter with the HDF5 library.  

Parameters
     [in] | cls | A pointer to a buffer for the struct containing the filter-definition  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.") registers a new filter with the HDF5 library.
Making a new filter available to an application is a two-step process. The first step is to write the three filter callback functions described below: `can_apply`, `set_local`, and `filter`. This call to [H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library."), registering the filter with the library, is the second step. The can_apply and set_local fields can be set to NULL if they are not required for the filter being registered.
[H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.") accepts a single parameter, a pointer to a buffer for the `cls` data structure. That data structure must conform to one of the following definitions: 
typedef struct [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html) {
[H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) [id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html#a6f3027958c7ad4552356a1e47e4db2ce); 
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html#a8f8f80d37794cde9472343e4487ba3eb); 
[H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) [can_apply](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html#a10d4a5417743a68aafc95160c5152601); 
[H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) [set_local](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html#a5d89ab9372734010a5c500ea3e694a03); 
[H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6) [filter](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html#a995507f1980f1e7f971fda7c362a213b); 
} [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html);
or 
typedef struct [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html) {
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#aad880fc4455c253781e8968f2239d56f); 
[H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) [id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a6f3027958c7ad4552356a1e47e4db2ce); 
unsigned [encoder_present](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a1622cd6083cfb9a6c4efb5d5bf67b143); 
unsigned [decoder_present](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#af3f8cbfa487cb4c06ad2210d0b81a271); 
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a8f8f80d37794cde9472343e4487ba3eb); 
[H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510) [can_apply](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a10d4a5417743a68aafc95160c5152601); 
[H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e) [set_local](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a5d89ab9372734010a5c500ea3e694a03); 
[H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6) [filter](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html#a995507f1980f1e7f971fda7c362a213b); 
} [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html);
`version` is a library-defined value reporting the version number of the [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) struct. This currently must be set to [H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9).
`id` is the identifier for the new filter. This is a user-defined value between [H5Z_FILTER_RESERVED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a5280cfd4cf81c09616958ce34547c15e) and [H5Z_FILTER_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ab47f5d3a9a56dad8db2014fa25eb3be0). These values are defined in the HDF5 source file [H5Zpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html), but the symbols [H5Z_FILTER_RESERVED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a5280cfd4cf81c09616958ce34547c15e) and [H5Z_FILTER_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#ab47f5d3a9a56dad8db2014fa25eb3be0) should always be used instead of the literal values.
`encoder_present` is a library-defined value indicating whether the filter's encoding capability is available to the application.
`decoder_present` is a library-defined value indicating whether the filter's encoding capability is available to the application.
`name` is a descriptive comment used for debugging, may contain a descriptive name for the filter, and may be the null pointer.
`can_apply`, described in detail below, is a user-defined callback function that determines whether the combination of the dataset creation property list values, the datatype, and the dataspace represent a valid combination to apply this filter to.
`set_local`, described in detail below, is a user-defined callback function that sets any parameters that are specific to this dataset, based on the combination of the dataset creation property list values, the datatype, and the dataspace.
`filter`, described in detail below, is a user-defined callback function which performs the action of the filter.
The statistics associated with a filter are not reset by this function; they accumulate over the life of the library.
[H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) is a macro that maps to either [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html) or [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html), depending on the needs of the application. To affect only this macro, H5Z_class_t_vers may be defined as either 1 or 2. Otherwise, it will behave in the same manner as other API compatibility macros. See [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html) for more information. [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html) matches the [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) structure that is used in the 1.6.x versions of the HDF5 library.
[H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.") will automatically detect which structure type has been passed in, regardless of the mapping of the [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) macro. However, the application must make sure that the fields are filled in according to the correct structure definition if the macro is used to declare the structure.
**The callback functions:**  
Before [H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.") can link a filter into an application, three callback functions must be defined as described in the HDF5 library header file [H5Zpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html).
When a filter is applied to the fractal heap for a group (e.g., when compressing group metadata) and if they can apply and set local callback functions that have been defined for that filter, HDF5 passes the value -1 for all parameters for those callback functions. This is done to ensure that the filter will not be applied to groups if it relies on these parameters, as they are not applicable to group fractal heaps; to operate on group fractal heaps, a filter must be capable of operating on an opaque block of binary data.
The _can-apply_ callback function must return a positive value for a valid combination, zero for an invalid combination, and a negative value for an error. 
typedef [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) (*[H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id);
Before a dataset is created, the _can apply_ callbacks for any filters used in the dataset creation property list are called with the dataset's dataset creation property list, `dcpl_id`, the dataset's datatype, `type_id`, and a dataspace describing a chunk, `space_id`, (for chunked dataset storage).
This callback must determine whether the combination of the dataset creation property list settings, the datatype, and the dataspace represent a valid combination to which to apply this filter. For example, an invalid combination may involve the filter not operating correctly on certain datatypes, on certain datatype sizes, or on certain sizes of the chunk dataspace. If this filter is enabled through [H5Pset_filter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") as optional and the can apply function returns 0, the library will skip the filter in the filter pipeline.
This callback can be the NULL pointer, in which case the library will assume that the filter can be applied to a dataset with any combination of dataset creation property list values, datatypes, and dataspaces.
The _set local_ callback function is defined as follows: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5Z_set_local_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a32591ae9c5164edd548c9885f430b15e))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id);
After the can apply callbacks are checked for a new dataset, the _set local_ callback functions for any filters used in the dataset creation property list are called. These callbacks receive `dcpl_id`, the dataset's private copy of the dataset creation property list passed into [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (i.e. not the actual property list passed into [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)); `type_id`, the datatype identifier passed into [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), which is not copied and should not be modified; and `space_id`, a dataspace describing the chunk (for chunked dataset storage), which should also not be modified.
The set local callback must set any filter parameters that are specific to this dataset, based on the combination of the dataset creation property list values, the datatype, and the dataspace. For example, some filters perform different actions based on different datatypes, datatype sizes, numbers of dimensions, or dataspace sizes.
The _set local_ callback may be the NULL pointer, in which case, the library will assume that there are no dataset-specific settings for this filter.
The _set local_ callback function must return a non-negative value on success and a negative value for an error.
The _filter operation_ callback function, defining the filter's operation on the data, is defined as follows: 
typedef [size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd) (*[H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6))(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
size_t nbytes, size_t *buf_size, void **buf);
The parameters `flags`, `cd_nelmts`, and `cd_values` are the same as for the function [H5Pset_filter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline."). The one exception is that an additional flag, [H5Z_FLAG_REVERSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a843859882318f69c9b4a38d3a88e0bfb), is set when the filter is called as part of the input pipeline.
The parameter `buf` points to the input buffer which has a size of `buf_size` bytes, `nbytes` of which are valid data.
The filter should perform the transformation in place if possible. If the transformation cannot be done in place, then the filter should allocate a new buffer and assign it to `buf`, assigning the allocated size of that buffer to `buf_size`. The old buffer should be freed by the filter.
Some care must be taken with the functions that allocate and free memory. Standard C library functions like malloc(3) and free(3) will work in many cases, but if there is a mismatch between the memory allocators used in the library and any filter that reallocates a buffer, there could be problems. This is most often the case with Windows and/or when debugging memory allocators are being used. In both cases, the "state" of the memory allocator lies in different libraries and will get corrupted if you allocate in one library and free in another. Windows adds the C standard library via dlls that can vary with Visual Studio version and debug vs. release builds. Static links to the MSVC CRT can also introduce a new memory allocator state.
The library does provide [H5allocate_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683 "Allocates memory that will be freed later internally.") and [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") functions that will use the library's allocation and free functions, however using these functions will require linking your filter to a particular version of the library, which may be inconvenient.
If successful, the _filter operation_ callback function returns the number of valid bytes of data contained in `buf`. In the case of failure, the return value is 0 (zero) and all pointer arguments are left unchanged. 

Version
    1.8.6 Return type for the _can apply_ callback function, [H5Z_can_apply_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#af2d1e20aeb92b2712ebc2d9b5fcbf510), changed to [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca).       1.8.5 Semantics of the _can apply_ and _set local_ callback functions changed to accommodate the use of filters with group fractal heaps.       1.8.3 [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) renamed to [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html), [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html) structure introduced for backwards compatibility with release 1.6.x, and [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) macro introduced in this release. Function modified to accept either structure type.       1.8.0 The fields `version`, `encoder_present`, and `decoder_present` were added to the [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) `struct` in this release.       1.6.0 This function was substantially revised in Release 1.6.0 with a new [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) struct and new set local and can apply callback functions. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8)H5Zunregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Zunregister  | ( | [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237) | _id_ | ) |   
---|---|---|---|---|---  
Unregisters a filter.  

Parameters
     [in] | id | Identifier of the filter to be unregistered.   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Zunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8 "Unregisters a filter.") unregisters the filter specified in `id`.
This function first iterates through all opened datasets and groups. If an open object that uses this filter is found, the function will fail with a message indicating that an object using the filter is still open. All open files are then flushed to make sure that all cached data that may use this filter are written out.
If the application is a parallel program, all processes that participate in collective data writing should call this function to ensure that all data is flushed.
After a call to [H5Zunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8 "Unregisters a filter."), the filter specified in filter will no longer be available to the application. 

Version
    1.8.12 Function modified to check for open objects using the filter.  

Since
    1.6.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


