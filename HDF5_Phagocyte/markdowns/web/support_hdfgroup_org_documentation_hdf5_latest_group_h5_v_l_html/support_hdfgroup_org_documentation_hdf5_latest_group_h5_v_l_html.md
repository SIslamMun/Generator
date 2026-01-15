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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title3 "H2")
  * [◆ H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title4 "H2")
  * [◆ H5VLget_connector_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title5 "H2")
  * [◆ H5VLget_connector_id_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title6 "H2")
  * [◆ H5VLget_connector_id_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title7 "H2")
  * [◆ H5VLget_connector_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title8 "H2")
  * [◆ H5VLis_connector_registered_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title9 "H2")
  * [◆ H5VLis_connector_registered_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title10 "H2")
  * [◆ H5VLobject_is_native()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title11 "H2")
  * [◆ H5VLquery_optional()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title12 "H2")
  * [◆ H5VLregister_connector_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title13 "H2")
  * [◆ H5VLregister_connector_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title14 "H2")
  * [◆ H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title15 "H2")
  * [◆ H5VLwrap_register()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#title16 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#func-members)
VOL connector (H5VL)
## Detailed Description
Use the functions in this module to manage HDF5 Virtual Object Layer (VOL) connectors.
VOL connectors provide an abstraction layer for storage operations, enabling HDF5 to work with different storage backends while maintaining the familiar HDF5 API. This module provides functions for registering, configuring, and managing VOL connector plugins.
### VOL Connector Life Cycle
VOL connectors can be implemented as dynamically loaded plugins, statically linked libraries, or built into the HDF5 library itself. Throughout this documentation, "VOL connector" refers to the connector implementation regardless of how it is deployed.
VOL connectors follow a well-defined life cycle from loading to cleanup:
  1. **Discovery and Loading**
     * For dynamically loaded plugins: The library searches the plugin path (default: /usr/local/hdf5/lib/plugin on POSIX, ALLUSERSPROFILE%/hdf5/lib/plugin on Windows) 
     * Plugin path can be overridden using the [HDF5_PLUGIN_PATH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a020d112b3c3251e7518461466777611d) environment variable 
     * For statically linked connectors: The connector is available immediately when the library loads 
     * For internal connectors: Built into the HDF5 library (e.g., the native VOL connector) 
  2. **Registration**
     * Before use, connectors must be registered with the library using: 
       * [H5VLregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b "Registers a new VOL connector.") - Registers a new VOL connector 
       * [H5VLregister_connector_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1 "Registers a new VOL connector by name.") - Register by connector name 
       * [H5VLregister_connector_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8 "Registers a new VOL connector by value.") - Register by connector-specific value 
     * Registration loads the connector and returns an HDF5 identifier ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) 
     * Many connectors provide initialization functions that handle registration automatically 
     * Registration can also occur automatically via the [HDF5_VOL_CONNECTOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8d21b0c065a27e1b6ea46efb966c7394) environment variable 
  3. **Configuration**
     * The connector is set in a file access property list (fapl) using: 
       * [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0 "Set the file VOL connector for a file access property list.") - Generic method using connector ID and optional configuration 
       * Connector-specific functions (e.g., H5Pset_fapl_<name>()) - Often more convenient 
       * [HDF5_VOL_CONNECTOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8d21b0c065a27e1b6ea46efb966c7394) environment variable - Sets the default connector for all file opens 
     * Configuration may include connector-specific parameters passed via info structs or parameter strings 
     * Parameter strings can be parsed using [H5VLconnector_str_to_info()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f)
  4. **Active Use**
     * When a file is opened or created with a configured fapl, the specified VOL connector handles all storage operations 
     * All API calls that manipulate storage (create, open, read, write, etc.) are forwarded to the connector's callbacks 
     * The connector remains active for the lifetime of all files opened with it 
     * Multiple different connectors can be active simultaneously for different files 
  5. **Query and Introspection**
     * [H5Pget_vol_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838 "Returns the identifier of the current VOL connector.") retrieves the connector ID from a fapl 
     * [H5Pget_vol_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f "Returns a copy of the VOL information for a connector.") retrieves connector-specific configuration information 
     * [H5VLget_connector_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa "Retrieves a connector name for a VOL.") retrieves the connector's registered name 
     * [H5VLis_connector_registered_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca "Tests whether a VOL class has been registered under a certain name.") checks if a connector is registered by name 
     * [H5VLis_connector_registered_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b "Tests whether a VOL class has been registered for a given value.") checks if a connector is registered by value 
     * [H5VLquery_optional()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043 "Determine if a VOL connector supports a particular optional callback operation.") determines support for optional operations 
  6. **Cleanup and Unregistration**
     * Connector IDs can be closed using: 
       * [H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e "Removes a VOL connector identifier from the library.") - Unregister a connector 
       * [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.") - Close a connector ID (same internal implementation) 
     * The library maintains reference counts on connector IDs and will not actually close them until the reference count reaches zero 
     * It is safe to close connector IDs after use, even while files opened with that connector remain open 
     * The library automatically unloads all connectors when it shuts down 
     * The native VOL connector cannot be unloaded and is always available 
     * Connector-specific info structures should be freed using [H5VLfree_connector_info()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d7c204a3db83d5563b0be557a3a4571)



See also
    H5VL_UG 
##  Topics  
---  
| [Definitions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html)  
| [Native VOL](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html)  
| [Pass-through VOL](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_p_t.html)  
| [VOL Developer](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id)  
| Closes a VOL connector identifier.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLget_connector_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga5b69c29931e55208517c598ac3039f77) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id)  
| Retrieves the VOL connector identifier for a given object identifier.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLget_connector_id_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gabcbf9b9b07a6b60e17ff9681684f944d) (const char *name)  
| Retrieves the identifier for a registered VOL connector name.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLget_connector_id_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee) ([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value)  
| Retrieves the identifier for a registered VOL connector value.   
  
ssize_t  |  [H5VLget_connector_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, char *name, size_t size)  
| Retrieves a connector name for a VOL.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5VLis_connector_registered_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca) (const char *name)  
| Tests whether a VOL class has been registered under a certain name.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5VLis_connector_registered_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b) ([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value)  
| Tests whether a VOL class has been registered for a given value.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLobject_is_native](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga7c9103049619e06acff40930a21d3caf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, bool *is_native)  
| Determines whether an object ID represents a native VOL connector object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLquery_optional](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, int opt_type, uint64_t *flags)  
| Determine if a VOL connector supports a particular optional callback operation.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1) (const char *connector_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id)  
| Registers a new VOL connector by name.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8) ([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id)  
| Registers a new VOL connector by value.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLunregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id)  
| Removes a VOL connector identifier from the library.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5VLwrap_register](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9873d50b395911b609621c22c2fa554b) (void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Wrap an internal object with a "wrap context" and register an [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) for the resulting object.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094)H5VLclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _connector_id_ | ) |   
---|---|---|---|---|---  
Closes a VOL connector identifier.  

Parameters
     [in] | connector_id | Connector identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.") closes a VOL connector identifier. This does not affect the file access property lists which have been defined to use this VOL connector or files which are already opened under this connector. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga5b69c29931e55208517c598ac3039f77)H5VLget_connector_id()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLget_connector_id  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _obj_id_ | ) |   
---|---|---|---|---|---  
Retrieves the VOL connector identifier for a given object identifier.  

Parameters
     [in] | obj_id | Object identifier   
---|---|--- 

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLget_connector_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga5b69c29931e55208517c598ac3039f77 "Retrieves the VOL connector identifier for a given object identifier.") retrieves the registered VOL connector identifier for the specified object identifier `obj_id`. The VOL connector identifier must be closed with [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.") when no longer in use. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gabcbf9b9b07a6b60e17ff9681684f944d)H5VLget_connector_id_by_name()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLget_connector_id_by_name  | ( | const char * | _name_ | ) |   
---|---|---|---|---|---  
Retrieves the identifier for a registered VOL connector name.  

Parameters
     [in] | name | Connector name   
---|---|--- 

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLget_connector_id_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gabcbf9b9b07a6b60e17ff9681684f944d "Retrieves the identifier for a registered VOL connector name.") retrieves the identifier for a registered VOL connector with the name `name`. The identifier must be closed with [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.") when no longer in use. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee)H5VLget_connector_id_by_value()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLget_connector_id_by_value  | ( | [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) | _connector_value_ | ) |   
---|---|---|---|---|---  
Retrieves the identifier for a registered VOL connector value.  

Parameters
     [in] | connector_value | Connector value   
---|---|--- 

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLget_connector_id_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee "Retrieves the identifier for a registered VOL connector value.") retrieves the identifier for a registered VOL connector with the value `connector_value`. The identifier will need to be closed by [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.").
`connector_value` has a type of [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c "VOL connector identifiers."), which is defined in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) as follows: 
typedef int [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c);
Valid VOL connector identifiers can have values from 0 through 255 for connectors defined by the HDF5 library. Values 256 through 511 are available for testing new connectors. Subsequent values should be obtained by contacting the The HDF Help Desk. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa)H5VLget_connector_name()
ssize_t H5VLget_connector_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _id_ ,   
---|---|---|---  
|  | char * |  _name_ ,   
|  | size_t |  _size_ )  
Retrieves a connector name for a VOL.  

Parameters
     [in] | id | Object identifier or file identifier   
---|---|---  
[out] | name | Connector name   
[in] | size | Maximum length of the name to retrieve  

Returns
    Returns the length of the connector name on success, and a negative value on failure.  
[H5VLget_connector_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa "Retrieves a connector name for a VOL.") retrieves up to `size` elements of the VOL name `name` associated with the object or file identifier `id`.
Passing in a NULL pointer for size will return the size of the connector name. This can be used to determine the size of the buffer to allocate for the name. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca)H5VLis_connector_registered_by_name()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5VLis_connector_registered_by_name  | ( | const char * | _name_ | ) |   
---|---|---|---|---|---  
Tests whether a VOL class has been registered under a certain name.  

Parameters
     [in] | name | Alleged name of connector   
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5VLis_connector_registered_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca "Tests whether a VOL class has been registered under a certain name.") tests whether a VOL class has been registered or not, according to the supplied connector name `name`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b)H5VLis_connector_registered_by_value()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5VLis_connector_registered_by_value  | ( | [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) | _connector_value_ | ) |   
---|---|---|---|---|---  
Tests whether a VOL class has been registered for a given value.  

Parameters
     [in] | connector_value | Connector value   
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5VLis_connector_registered_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b "Tests whether a VOL class has been registered for a given value.") tests whether a VOL class has been registered, according to the supplied connector value `connector_value`.
`connector_value` has a type of [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c "VOL connector identifiers."), which is defined in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) as follows: 
typedef int [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c);
Valid VOL connector identifiers can have values from 0 through 255 for connectors defined by the HDF5 library. Values 256 through 511 are available for testing new connectors. Subsequent values should be obtained by contacting the The HDF Help Desk. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga7c9103049619e06acff40930a21d3caf)H5VLobject_is_native()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLobject_is_native  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | bool * |  _is_native_ )  
Determines whether an object ID represents a native VOL connector object.  

Parameters
     [in] | obj_id | Object identifier   
---|---|---  
[out] | is_native | Boolean determining whether object is a native VOL connector object  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Since
    1.12.2   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043)H5VLquery_optional()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLquery_optional  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) |  _subcls_ ,   
|  | int |  _opt_type_ ,   
|  | uint64_t * |  _flags_ )  
Determine if a VOL connector supports a particular optional callback operation.  

Parameters
     [in] | obj_id | Object identifier   
---|---|---  
[in] | subcls | VOL subclass   
[in] | opt_type | Option type   
[out] | flags | Operation flags  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Since
    1.12.1   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1)H5VLregister_connector_by_name()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLregister_connector_by_name  | ( | const char * |  _connector_name_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _vipl_id_ )  
Registers a new VOL connector by name.  

Parameters
     [in] | connector_name | Connector name   
---|---|---  
[in] | vipl_id | VOL initialization property list identifier  

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLregister_connector_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1 "Registers a new VOL connector by name.") registers a new VOL connector with the name `connector_name` as a member of the virtual object layer class. This VOL connector identifier is good until the library is closed or the connector is unregistered.
`vipl_id` is either [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) or the identifier of a VOL initialization property list of class [H5P_VOL_INITIALIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afebc2bfbcba7288957a33837b6a070a5) created with [H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class."). When created, this property list contains no library properties. If a VOL connector author decides that initialization-specific data are needed, they can be added to the empty list and retrieved by the connector in the VOL connector's initialize callback. Use of the VOL initialization property list is uncommon, as most VOL-specific properties are added to the file access property list via the connector's API calls which set the VOL connector for the file open/create. For more information, see VOL documentation. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8)H5VLregister_connector_by_value()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLregister_connector_by_value  | ( | [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) |  _connector_value_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _vipl_id_ )  
Registers a new VOL connector by value.  

Parameters
     [in] | connector_value | Connector value   
---|---|---  
[in] | vipl_id | VOL initialization property list identifier  

Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5VLregister_connector_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8 "Registers a new VOL connector by value.") registers a new VOL connector with value connector_value as a member of the virtual object layer class. This VOL connector identifier is good until the library is closed or the connector is unregistered.
`connector_value` has a type of [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c "VOL connector identifiers."), which is defined in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) as follows: 
typedef int [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c);
Valid VOL connector identifiers can have values from 0 through 255 for connectors defined by the HDF5 library. Values 256 through 511 are available for testing new connectors. Subsequent values should be obtained by contacting the The HDF Help Desk.
`vipl_id` is either [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) or the identifier of a VOL initialization property list of class [H5P_VOL_INITIALIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afebc2bfbcba7288957a33837b6a070a5) created with [H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class."). When created, this property list contains no library properties. If a VOL connector author decides that initialization-specific data are needed, they can be added to the empty list and retrieved by the connector in the VOL connector's initialize callback. Use of the VOL initialization property list is uncommon, as most VOL-specific properties are added to the file access property list via the connector's API calls which set the VOL connector for the file open/create. For more information, see the VOL documentation. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e)H5VLunregister_connector()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLunregister_connector  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _connector_id_ | ) |   
---|---|---|---|---|---  
Removes a VOL connector identifier from the library.  

Parameters
     [in] | connector_id | Connector identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e "Removes a VOL connector identifier from the library.") removes a VOL connector identifier from the library. This does not affect the file access property lists which have been defined to use the VOL connector or any files which are already opened with this connector. 

Attention
     [H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e "Removes a VOL connector identifier from the library.") will fail if attempting to unregister the native VOL connector. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9873d50b395911b609621c22c2fa554b)H5VLwrap_register()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5VLwrap_register  | ( | void * |  _obj_ ,   
---|---|---|---  
|  | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ )  
Wrap an internal object with a "wrap context" and register an [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) for the resulting object.  

Parameters
     [in] | obj | VOL object.   
---|---|---  
[in] | type | VOL-managed object class. Allowable values are:
  * [H5I_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bacc572b5478629d17dd4fa708c3508f22)
  * [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6)
  * [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64)
  * [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987)
  * [H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d)
  * [H5I_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5bdc68e9f466027aeac5f8b11205e51f)



Returns
    Returns a VOL connector identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). 

Note
    This routine is mainly targeted toward wrapping objects for iteration routine callbacks (i.e. the callbacks from H5Aiterate*, H5Literate* / H5Lvisit*, and H5Ovisit* ). Using it in an application will return an error indicating the API context isn't available or can't be retrieved.   
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


