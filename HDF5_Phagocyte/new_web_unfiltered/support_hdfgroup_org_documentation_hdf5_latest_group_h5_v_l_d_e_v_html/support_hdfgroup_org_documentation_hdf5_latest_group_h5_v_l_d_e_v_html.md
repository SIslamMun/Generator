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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title0 "H2")
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title3 "H2")
  * [◆ H5VLclose_lib_context()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title4 "H2")
  * [◆ H5VLget_file_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title5 "H2")
  * [◆ H5VLobject()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title6 "H2")
  * [◆ H5VLopen_lib_context()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title7 "H2")
  * [◆ H5VLregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#nested-classes) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#func-members)
VOL Developer
[VOL connector (H5VL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html)
## Detailed Description
##  Data Structures  
---  
struct  | [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLclose_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae05f169607f8be714fd34424c3786ee0) (void *context)  
| Closes the internal state of the HDF5 library.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLget_file_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga161553978d3d001a5b04708acccb429f) (void *file_obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id)  
void *  |  [H5VLobject](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga21f351a8a3a128659f57217a3b452cd5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLopen_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d) (void **context)  
| Opens a new internal context for the HDF5 library.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b) (const [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) *cls, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id)  
| Registers a new VOL connector.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae05f169607f8be714fd34424c3786ee0)H5VLclose_lib_context()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLclose_lib_context  | ( | void * | _context_ | ) |   
---|---|---|---|---|---  
Closes the internal state of the HDF5 library.  

Parameters
     [in] | context | library's context  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Closes the internal state of the HDF5 library, undoing the effects of [H5VLopen_lib_context()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d "Opens a new internal context for the HDF5 library."). This function must be called as a _pair_ with [H5VLopen_lib_context()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d "Opens a new internal context for the HDF5 library."). It can be invoked before, after, or independently of [H5VLfree_lib_state()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a37856e5f0914916c4ac36cbc0e6bed84). 

Note
    This routine is exclusively for authors of HDF5 VOL connectors. It is not part of the public HDF5 APIs for HDF5 application developers. 

Since
    2.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga161553978d3d001a5b04708acccb429f)H5VLget_file_type()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLget_file_type  | ( | void * |  _file_obj_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _connector_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dtype_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga21f351a8a3a128659f57217a3b452cd5)H5VLobject()
void * H5VLobject  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _obj_id_ | ) |   
---|---|---|---|---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d)H5VLopen_lib_context()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLopen_lib_context  | ( | void ** | _context_ | ) |   
---|---|---|---|---|---  
Opens a new internal context for the HDF5 library.  

Parameters
     [out] | context | library's context  
---|---|--- 

Returns
    Returns a non-negative value if `*context` is set; otherwise returns negative value if `*context` is unset.  
Opens a new internal context for the HDF5 library. The context returned (via the OUT parameter) must be passed to [H5VLclose_lib_context()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae05f169607f8be714fd34424c3786ee0 "Closes the internal state of the HDF5 library.") to conclude the library's context and release resources. 

Note
    This routine is exclusively for authors of HDF5 VOL connectors. It is not part of the public HDF5 APIs for HDF5 application developers. 

Since
    2.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b)H5VLregister_connector()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLregister_connector  | ( | const [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) * |  _cls_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _vipl_id_ )  
Registers a new VOL connector.  

Parameters
     [in] | cls | A pointer to the plugin structure to register   
---|---|---  
[in] | vipl_id | VOL initialization property list identifier  

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b "Registers a new VOL connector.") registers a new VOL connector as a member of the virtual object layer class. This VOL connector identifier is good until the library is closed or the connector is unregistered.
`vipl_id` is either [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) or the identifier of a VOL initialization property list of class [H5P_VOL_INITIALIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afebc2bfbcba7288957a33837b6a070a5) created with [H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class."). When created, this property list contains no library properties. If a VOL connector author decides that initialization-specific data are needed, they can be added to the empty list and retrieved by the connector in the VOL connector's initialize callback. Use of the VOL initialization property list is uncommon, as most VOL-specific properties are added to the file access property list via the connector's API calls which set the VOL connector for the file open/create. For more information, see the VOL documentation.
[H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) is defined in [H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html) in the source code. It contains class information for each VOL connector: 
typedef struct [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) {
/* Overall connector fields & callbacks */
unsigned [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a27fe8e82cf2ce359010cc1d08af2e105); 
[H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) [value](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a4531586e75e64de257cf51b4b7c50d70); 
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a8f8f80d37794cde9472343e4487ba3eb); 
unsigned [conn_version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a6566afc54591b6c64dbec7bdc0bf0b58); 
uint64_t [cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#ac4b405cf0a53f734d273232a82baf247); 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[initialize](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a0a18753f3a24e91b75fd802b81f044a7))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id); 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[terminate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a3da863f4634c64aeffe46cca882db43d))(void); 
/* VOL framework */
H5VL_info_class_t [info_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#afdbe51576a0f6360693757d6738111eb); 
H5VL_wrap_class_t [wrap_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a74d541c08c6c71946cc5a1e105ca5e51); 
/* Data Model */
H5VL_attr_class_t [attr_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aea95f1ef7bfe9d0a06f5d7ba4d992b33); 
H5VL_dataset_class_t [dataset_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a38be37dc57b556f87df46b0ea65c9ae0); 
H5VL_datatype_class_t [datatype_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a49a73ac039550d87d60b485727149cce); 
H5VL_file_class_t [file_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a94fcae0c3e55fb9da64b99cf5ec0cc05); 
H5VL_group_class_t [group_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a6f9e6afe57118c01efbd5805d6f6615d); 
H5VL_link_class_t [link_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a3da742717813fedde17f3ef9a739a3a5); 
H5VL_object_class_t [object_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aee15298970b7178f78f92de98b7f8211); 
/* Infrastructure / Services */
H5VL_introspect_class_t [introspect_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aff81b23d1373b396d5eacda97bb4a6c6); 
H5VL_request_class_t [request_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#adc5afcdccffe9dc3608ac8e23e8fea17); 
H5VL_blob_class_t [blob_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aee571d79753251a1d6ebc1073254278c); 
H5VL_token_class_t [token_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#acd39460513ffed8d7d3eab4ebab54bb2); 
/* Catch-all */
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, H5VL_optional_args_t *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req); 
} [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html); 

Since
    1.12.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


