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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title0 "H2")
  * [ Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title3 "H2")
  * [◆ H5LT_FILE_IMAGE_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title4 "H2")
  * [◆ H5LT_FILE_IMAGE_DONT_COPY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title5 "H2")
  * [◆ H5LT_FILE_IMAGE_DONT_RELEASE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title6 "H2")
  * [◆ H5LT_FILE_IMAGE_OPEN_RW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title7 "H2")
  * [Enumeration Type Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title8 "H2")
  * [◆ H5LT_lang_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#define-members) | [Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#enum-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#func-members)
H5LTpublic.h File Reference
##  Macros  
---  
#define  |  [H5LT_FILE_IMAGE_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a1462912209df9c6e0b29fa2a59b229a8) 0x0007  
#define  |  [H5LT_FILE_IMAGE_DONT_COPY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a9f2c5bfc75d6c23f20c3a303934c20a8) 0x0002 /* The HDF5 lib won't copy */  
#define  |  [H5LT_FILE_IMAGE_DONT_RELEASE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a6765cc939c4a8e1d9630f09bf177a761) 0x0004 /* The HDF5 lib won't */  
#define  |  [H5LT_FILE_IMAGE_OPEN_RW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#adb7efc091461c62bf5ab67a30c8fded6) 0x0001 /* Open image for read-write */  
##  Enumerations  
---  
enum  |  [H5LT_lang_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63) {   
[H5LT_LANG_ERR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a304e487ca906933388da7d3ab579736d) = -1 , [H5LT_DDL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a158bbaa59144a89547203f8e95421911) = 0 , [H5LT_C](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a9d8ebd69963c037728cee4539ef3bab7) = 1 , [H5LT_FORTRAN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63af78ba88ea06e2b952c37b4a213da1184) = 2 ,   
[H5LT_NO_LANG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a8fdcdb138e9136afa498eec8e344e8c7) = 3   
}  
##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTdtype_to_text](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga155a5e59441ed05ca1b91e6404314862) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype, char *str, [H5LT_lang_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63) lang_type, size_t *len)  
| Creates a text description of an HDF5 datatype.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTfind_attribute](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga4bc3a9d7a292eb02761c404eac05f8ac) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name)  
| Determines whether an attribute exists.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTfind_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga4523d5d0a683ed022a184209d5ce2927) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name)  
| Determines whether a dataset exists.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga85773540c703d4d1a29874a0aeae0ea9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, void *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_char](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga8016ebf96d60407ef0e955f3ceeb08e6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, char *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_double](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga385d4f2d4c0b7090cc782deb7a6a75c8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, double *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_float](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gabfde7457326ff08631c8662e68330dce) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, float *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga8111f14c59f7b72c73803442b1b42ff7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, [H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) *type_class, size_t *type_size)  
| Gets information about an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga26785bc328ff30f2217547c17fe2f6de) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, int *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga067ad008cca4f58f9814b137b2d343c0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, long *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_long_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga9849ee57e1a62ea78edceeb9e7eb4280) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, long long *data)  
| Reads a _long_ _long_ attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad20a9cf20f1de48602a441c6a7dcc8d7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, int *rank)  
| Gets the dimensionality of an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_short](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga765a091986bb0b45ddedde473cf6c915) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, short *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gab70558f8df13ff38fb78b79db047cea3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, char *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_uchar](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga589cfc1261a2eef2a6488c2e29dedac7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, unsigned char *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_uint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae117a5065a68268b800dc7c67bcf60fe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, unsigned int *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_ullong](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga090c3e4f719a5d60bf1d4b8961109371) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, unsigned long long *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_ulong](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga4a1a7a46b4a9bfe98f510a3132940854) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, unsigned long *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_attribute_ushort](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gac2bb9376730582f0a091d27a98f9b57a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, unsigned short *data)  
| Reads an attribute from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_dataset_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga608f1a76153de414722b75cf7831cca8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, [H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) *type_class, size_t *type_size)  
| Retrieves information about a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTget_dataset_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gaf93283cb49cccae513181caf15e8ed28) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int *rank)  
| Gets the dimensionality of a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad6db7625a400b605ec5a4ce12f00be0b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const void *buffer)  
| Creates and writes a dataset of a type `type_id`.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_char](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga2b20a9bfb57a2405baf278da52fffc06) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const char *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_double](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gaaeea3e0d0e7ead3dfac627cd60085f9d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const double *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_float](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga08fa486a415b87a78d32fbff6a74cbfb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const float *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga28891f2fbec407534394a8148c3f0908) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const int *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad03999f8e1b9576e8ea5dc2552ba936d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const long *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_short](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga656a2c0f71a84c48ce63cef70072f3f3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int rank, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims, const short *buffer)  
| Creates and writes a dataset.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTmake_dataset_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga0e3c5c63b9fbfce686b7c250f7a038a6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, const char *buf)  
| Creates and writes a dataset with string datatype.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5LTopen_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae2834cca05d6c4787f51933cc521f8fb) (void *buf_ptr, size_t buf_size, unsigned flags)  
| Opens an HDF5 file image in memory.   
  
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5LTpath_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga39038c92a8dcbe7bb542661572112089) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *path, bool check_object_valid)  
| Determines whether an HDF5 path is valid and, optionally, whether the path resolves to an HDF5 object.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gadf06e39ac477b18f34c351fd40a8c9fd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, void *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_char](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga24b290a7be0ccc3af7049e18f39a83f3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, char *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_double](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga32663a55239eee94d605bfa0d05c6716) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, double *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_float](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga5de60bee8f27cc929d292a9dff699c6b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, float *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga27dff339f933e51aaf0cf9169bed3256) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, int *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gabd8f411c596ed340b7ffcdac3fb717db) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, long *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_short](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gadb841e67f045f77222e4a8a8fd7d32f4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, short *buffer)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_dataset_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga599d6965ef34b39ac0fc9b965a4323dd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, char *buf)  
| Reads a dataset from disk.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_char](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga8012e9e47f2211a4fcfc2d39f653539d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const char *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_double](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga11210f9ad63c793f31c40426dd7d82c5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const double *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_float](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga1951f2234dc438e3dafcfb643d6576d0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const float *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga9b2c7d5eef5a1d3c90c6857ca337f737) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const int *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gaa1ccbc68afc9ad5851bb1f4ed7b305cc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const long *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_long_long](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga02d64d54eee3a22b257c3cf8508b9ae3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const long long *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_short](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga7509d53a6b727d9196d90b75564c6ab5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const short *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae60e8867e35c255dc5b5f3de134df80c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const char *attr_data)  
| Creates and writes a string attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_uchar](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga98004212f030c6c7f3536369954f9d88) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const unsigned char *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_uint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga685b20b1e24babbb20fe0a0fa0111d5d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const unsigned int *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_ullong](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga52a1e4f75326987651d75284d457d9e4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const unsigned long long *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_ulong](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gaf5820ae4a412abc5f872bef26bbd9051) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const unsigned long *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTset_attribute_ushort](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga1a8189cbd96bab4e213c144c26a530fa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, const unsigned short *buffer, size_t size)  
| Creates and writes an attribute.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f) (const char *text, [H5LT_lang_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63) lang_type)  
| Creates an HDF5 datatype given a text description.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a1462912209df9c6e0b29fa2a59b229a8)H5LT_FILE_IMAGE_ALL
#define H5LT_FILE_IMAGE_ALL 0x0007  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a9f2c5bfc75d6c23f20c3a303934c20a8)H5LT_FILE_IMAGE_DONT_COPY
#define H5LT_FILE_IMAGE_DONT_COPY 0x0002 /* The HDF5 lib won't copy */  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#a6765cc939c4a8e1d9630f09bf177a761)H5LT_FILE_IMAGE_DONT_RELEASE
#define H5LT_FILE_IMAGE_DONT_RELEASE 0x0004 /* The HDF5 lib won't */  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#adb7efc091461c62bf5ab67a30c8fded6)H5LT_FILE_IMAGE_OPEN_RW
#define H5LT_FILE_IMAGE_OPEN_RW 0x0001 /* Open image for read-write */  
---  
## Enumeration Type Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63)H5LT_lang_t
enum [H5LT_lang_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63)  
---  
Enumerator  
---  
H5LT_LANG_ERR  |   
H5LT_DDL  |   
H5LT_C  |   
H5LT_FORTRAN  |   
H5LT_NO_LANG  |   
  * [hl](https://support.hdfgroup.org/documentation/hdf5/latest/dir_12c0298e2ebcdf123b0eea2eebfe38f1.html)
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_581b2952518ba9b2ff64b9c38dd08f76.html)
  * [H5LTpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


