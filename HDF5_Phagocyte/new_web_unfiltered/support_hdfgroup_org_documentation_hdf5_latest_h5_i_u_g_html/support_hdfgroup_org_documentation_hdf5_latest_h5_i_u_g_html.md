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
  * [ The HDF5 Identifier Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title1 "H2")
  * [ Identifier Types](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title2 "H2")
  * [ Reference Counting](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title3 "H2")
  * [ Identifier Validation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title4 "H2")
  * [ Querying Identifiers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title5 "H2")
  * [ User-Defined Identifier Types](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title6 "H2")
  * [ Identifier Iteration](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title7 "H2")
  * [ Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i__u_g.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Identifiers
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Identifier Interface
##  Introduction
The HDF5 Identifier interface (H5I) manages identifiers (type [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) that serve as handles to HDF5 objects and resources. Identifiers provide an abstraction layer between applications and the internal HDF5 library structures, enabling safe and efficient object management.
Every HDF5 object—file, group, dataset, datatype, dataspace, attribute, property list—is accessed through an identifier. The H5I interface provides functions to query, validate, and manage these identifiers and their associated resources.
##  Identifier Types
HDF5 defines several built-in identifier types:
  * [H5I_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bacc572b5478629d17dd4fa708c3508f22) - File identifiers 
  * [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6) - Group identifiers 
  * [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64) - Datatype identifiers 
  * [H5I_DATASPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bade3bcb0953b041371997f802fa678da8) - Dataspace identifiers 
  * [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987) - Dataset identifiers 
  * [H5I_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5bdc68e9f466027aeac5f8b11205e51f) - Attribute identifiers 
  * [H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d) - Map identifiers 
  * [H5I_VFL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba0d3b691d8e02ae4898c82535401bee05) - Virtual File Layer driver identifiers 
  * [H5I_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bae950f33b1244e71d24b16786964f04b9) - Virtual Object Layer connector identifiers 
  * Property list identifiers (various classes)


Each identifier type is managed independently with its own reference counting system.
##  Reference Counting
Identifiers use reference counting to manage object lifetimes. When an identifier is created, its reference count is set to 1. The reference count can be manipulated to control when resources are released:
  * [H5Iget_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gac9638ade14cc75b7b125b3723f319c81 "Retrieves the reference count for an object.") retrieves the current reference count 
  * [H5Iinc_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f "Increments the reference count for an object.") increments the reference count 
  * [H5Idec_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9 "Decrements the reference count for an object.") decrements the reference count


When a reference count reaches zero, the associated resources are automatically released.
##  Identifier Validation
The H5I interface provides functions to validate identifiers:
  * [H5Iis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a "Determines whether an identifier is valid.") checks if an identifier is valid and has a positive reference count 
  * [H5Iget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d "Retrieves the type of an object.") retrieves the type of an identifier 
  * [H5Itype_exists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gac137a599a2055909918e55955a69dfad "Determines whether an identifier type is registered.") checks if an identifier type is registered


##  Querying Identifiers
Applications can retrieve information about objects through their identifiers:
  * [H5Iget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") retrieves the name of an object (for files, groups, datasets, etc.) 
  * [H5Iget_file_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042 "Retrieves an identifier for the file containing the specified object.") retrieves the file identifier associated with any object


These functions are particularly useful for debugging and logging.
##  User-Defined Identifier Types
Advanced applications can register custom identifier types using [H5Iregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it."). This allows user-defined objects to benefit from HDF5's identifier management:
  * Automatic reference counting 
  * Type safety through identifier types 
  * Integration with HDF5's resource management


Custom identifier types can be destroyed using [H5Idestroy_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e "Removes an identifier type and all identifiers within that type.") when no longer needed.
##  Identifier Iteration
The H5I interface allows iteration over all identifiers of a given type:
  * [H5Iiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4 "Calls a callback for each member of the identifier type specified.") iterates over identifiers of a specified type with a callback function 
  * [H5Inmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga839170c17fb8be7e14d16ee74a314a33 "Returns the number of identifiers in a given identifier type.") returns the number of identifiers of a given type


This is useful for resource tracking, debugging, and cleanup operations.
##  Summary
The H5I identifier interface provides essential functionality: 
  * Safe handles to HDF5 objects and resources 
  * Reference counting for automatic resource management 
  * Type checking and validation 
  * Object information retrieval 
  * Support for user-defined identifier types 
  * Iteration capabilities for resource tracking


Proper identifier management is fundamental to writing robust HDF5 applications that efficiently manage memory and avoid resource leaks.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


