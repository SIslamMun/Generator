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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title1 "H2")
  * [ Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title2 "H2")
  * [ Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title3 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title4 "H2")
  * [◆ H5VL_OPT_QUERY_SUPPORTED](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title5 "H2")
  * [◆ H5VL_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title6 "H2")
  * [Typedef Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title7 "H2")
  * [◆ H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title8 "H2")
  * [Enumeration Type Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title9 "H2")
  * [◆ H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#title10 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#define-members) | [Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#typedef-members) | [Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#enum-members)
Definitions
[VOL connector (H5VL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html)
## Detailed Description
##  Macros  
---  
#define  |  [H5VL_OPT_QUERY_SUPPORTED](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga2208d2bf3252e8201b80d48d9bfdd26c) 0x0001  
#define  |  [H5VL_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga2612dc1852fe3f855c755be6e058d56a) 3  
| Version number of VOL class struct & callbacks.   
  
##  Typedefs  
---  
typedef int  | [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c)  
| VOL connector identifiers.   
  
##  Enumerations  
---  
enum  |  [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) {   
[H5VL_SUBCLS_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac05e9c424f4c57ab04bb8a0f27680765) , [H5VL_SUBCLS_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6382b83356def3b14a27c06488a46b62) , [H5VL_SUBCLS_WRAP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a834b756ff01e1edb2979a0be92c3518b) , [H5VL_SUBCLS_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac49cca2c43d9a93d28ded5b9dc9a14d1) ,   
[H5VL_SUBCLS_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac9b25c8d8ea5205bf8f0fb654d29a57b) , [H5VL_SUBCLS_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6f3df58583ad3a02b32f1ea9e9a233c7) , [H5VL_SUBCLS_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a8611b474759b782775d303acd28c512f) , [H5VL_SUBCLS_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599aeb711151c2a908ae42e18e80bb7a8f1d) ,   
[H5VL_SUBCLS_LINK](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6d1475a46f9db62a48b9362362016e83) , [H5VL_SUBCLS_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a82eb9b3d6f086cafcb446eb86534a813) , [H5VL_SUBCLS_REQUEST](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ad23e1011cd67c7280a90ec903f210c08) , [H5VL_SUBCLS_BLOB](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599af6ba173c8e6b99d3df13f26d4f943e66) ,   
[H5VL_SUBCLS_TOKEN](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a3a4313ed710d7a5a5dbfb9ccc354c8ac)   
}  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga2208d2bf3252e8201b80d48d9bfdd26c)H5VL_OPT_QUERY_SUPPORTED
#define H5VL_OPT_QUERY_SUPPORTED 0x0001  
---  
Flags to return from H5VLquery_optional API and 'opt_query' callbacks
Operations which access multiple objects' data or metadata in a container should be registered as file-level optional operations. (e.g. "H5Dwrite_multi" takes a list of datasets to write data to, so a VOL connector that implemented it should register it as an optional file operation, and pass-through VOL connectors that are stacked above the connector that registered it should assume that dataset elements for _any_ dataset in the file could be written to) VOL connector supports this operation  

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga2612dc1852fe3f855c755be6e058d56a)H5VL_VERSION
#define H5VL_VERSION 3  
---  
Version number of VOL class struct & callbacks. 
Each VOL connector must set the 'version' field in the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct to the version of the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct that the connector implements. The HDF5 library will reject connectors with incompatible structs. 
## Typedef Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c)H5VL_class_value_t
typedef int [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c)  
---  
VOL connector identifiers. 
Values 0 through 255 are for connectors defined by the HDF5 library. Values 256 through 511 are available for testing new connectors. Subsequent values should be obtained from the HDF5 development team at [help@.nosp@m.hdfg.nosp@m.roup..nosp@m.org](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html). 
## Enumeration Type Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599)H5VL_subclass_t
enum [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599)  
---  
Enum type for each VOL subclass (Used for various queries, etc) 
Enumerator  
---  
H5VL_SUBCLS_NONE  |  Operations outside of a subclass   
H5VL_SUBCLS_INFO  |  'Info' subclass   
H5VL_SUBCLS_WRAP  |  'Wrap' subclass   
H5VL_SUBCLS_ATTR  |  'Attribute' subclass   
H5VL_SUBCLS_DATASET  |  'Dataset' subclass   
H5VL_SUBCLS_DATATYPE  |  'Named datatype' subclass   
H5VL_SUBCLS_FILE  |  'File' subclass   
H5VL_SUBCLS_GROUP  |  'Group' subclass   
H5VL_SUBCLS_LINK  |  'Link' subclass   
H5VL_SUBCLS_OBJECT  |  'Object' subclass   
H5VL_SUBCLS_REQUEST  |  'Request' subclass   
H5VL_SUBCLS_BLOB  |  'Blob' subclass   
H5VL_SUBCLS_TOKEN  |  'Token' subclass   
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


