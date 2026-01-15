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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title2 "H2")
  * [◆ H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title3 "H2")
  * [◆ H5VLcmp_connector_cls()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title4 "H2")
  * [◆ H5VLget_connector_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title5 "H2")
  * [◆ H5VLget_connector_id_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title6 "H2")
  * [◆ H5VLget_connector_id_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title7 "H2")
  * [◆ H5VLget_connector_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title8 "H2")
  * [◆ H5VLis_connector_registered_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title9 "H2")
  * [◆ H5VLis_connector_registered_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title10 "H2")
  * [◆ H5VLregister_connector_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title11 "H2")
  * [◆ H5VLregister_connector_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title12 "H2")
  * [◆ H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#title13 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#func-members)
Java VOL Connector (H5VL) Interface
## Detailed Description 

See also
     [VOL connector (H5VL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html), C-API      [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html), User Guide 
##  Functions  
---  
static synchronized native void  |  [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga7eb33aa9608a32287769a5d797c62b54) (long connector_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5VLcmp_connector_cls](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga6aeaace8a5926058662fbe1b6579ad6f) (long conn_id1, long conn_id2) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5VLget_connector_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga24d2122ae620f948e576430a2e6adec0) (long object_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5VLget_connector_id_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga192901b47f43b4dc00f7bbef0f308a2f) (String name) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5VLget_connector_id_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga5a90545b82cc2551d32eaae12059197e) (int connector_value) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native String  |  [H5VLget_connector_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga8130a8e4974f9bb3e8491cd30581cf93) (long object_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5VLis_connector_registered_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga54d7cb0f2b8c9ad464b010c4ccad6df4) (String name) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5VLis_connector_registered_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga05c5ba20915d9a493474b47c5d49832e) (int connector_value) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga501c72a5ce21c79f4803600965789b9f) (String connector_name, long vipl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga47a607e0d5770b1c79591f9c9c594fad) (int connector_value, long vipl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5VLunregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#gaec02be3715f1a4d71ba1a5d78524cbb3) (long connector_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga7eb33aa9608a32287769a5d797c62b54)H5VLclose()
| static synchronized native void H5VLclose  | ( | long | _connector_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLclose closes a VOL connector ID. 

Parameters
     connector_id | IN: Identifier of the connector.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga6aeaace8a5926058662fbe1b6579ad6f)H5VLcmp_connector_cls()
| static synchronized native boolean H5VLcmp_connector_cls  | ( | long |  _conn_id1_ ,   
---|---|---|---  
|  | long |  _conn_id2_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5VLcmp_connector_cls Determines whether two connector identifiers refer to the same connector. 

Parameters
     conn_id1 | IN: Identifier of connector to compare.   
---|---  
conn_id2 | IN: Identifier of connector to compare. 

Returns
    true if the connector identifiers refer to the same connector, else false. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga24d2122ae620f948e576430a2e6adec0)H5VLget_connector_id()
| static synchronized native long H5VLget_connector_id  | ( | long | _object_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLget_connector_id retrieves the ID for a registered VOL connector for a given object. 

Parameters
     object_id | IN: Identifier of the object.  
---|--- 

Returns
    a VOL connector ID 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga192901b47f43b4dc00f7bbef0f308a2f)H5VLget_connector_id_by_name()
| static synchronized native long H5VLget_connector_id_by_name  | ( | String | _name_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLget_connector_id_by_name retrieves the ID for a registered VOL connector. 

Parameters
     name | IN: name of the connector.  
---|--- 

Returns
    a VOL connector ID 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga5a90545b82cc2551d32eaae12059197e)H5VLget_connector_id_by_value()
| static synchronized native long H5VLget_connector_id_by_value  | ( | int | _connector_value_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLget_connector_id_by_value retrieves the ID for a registered VOL connector. 

Parameters
     connector_value | IN: value of the connector.  
---|--- 

Returns
    a VOL connector ID 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga8130a8e4974f9bb3e8491cd30581cf93)H5VLget_connector_name()
| static synchronized native String H5VLget_connector_name  | ( | long | _object_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLget_connector_name returns the connector name for the VOL associated with the object or file ID. 

Parameters
     object_id | IN: Identifier of the object.  
---|--- 

Returns
    the connector name 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga54d7cb0f2b8c9ad464b010c4ccad6df4)H5VLis_connector_registered_by_name()
| static synchronized native boolean H5VLis_connector_registered_by_name  | ( | String | _name_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLis_connector_registered_by_name tests whether a VOL class has been registered. 

Parameters
     name | IN: name of the connector.  
---|--- 

Returns
    true if a VOL connector with that name has been registered 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga05c5ba20915d9a493474b47c5d49832e)H5VLis_connector_registered_by_value()
| static synchronized native boolean H5VLis_connector_registered_by_value  | ( | int | _connector_value_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLis_connector_registered_by_value tests whether a VOL class has been registered. 

Parameters
     connector_value | IN: value of the connector.  
---|--- 

Returns
    true if a VOL connector with that value has been registered 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga501c72a5ce21c79f4803600965789b9f)H5VLregister_connector_by_name()
| static synchronized native long H5VLregister_connector_by_name  | ( | String |  _connector_name_ ,   
---|---|---|---  
|  | long |  _vipl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5VLregister_connector_by_name registers a new VOL connector as a member of the virtual object layer class. 

Parameters
     connector_name | IN: name of the connector.   
---|---  
vipl_id | IN: VOL initialization property list which must be created with H5Pcreate(H5P_VOL_INITIALIZE) (or H5P_DEFAULT). 

Returns
    a VOL connector ID 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#ga47a607e0d5770b1c79591f9c9c594fad)H5VLregister_connector_by_value()
| static synchronized native long H5VLregister_connector_by_value  | ( | int |  _connector_value_ ,   
---|---|---|---  
|  | long |  _vipl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5VLregister_connector_by_value registers a new VOL connector as a member of the virtual object layer class. 

Parameters
     connector_value | IN: value of the connector.   
---|---  
vipl_id | IN: VOL initialization property list which must be created with H5Pcreate(H5P_VOL_INITIALIZE) (or H5P_DEFAULT). 

Returns
    a VOL connector ID 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html#gaec02be3715f1a4d71ba1a5d78524cbb3)H5VLunregister_connector()
| static synchronized native void H5VLunregister_connector  | ( | long | _connector_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5VLunregister_connector removes a VOL connector ID from the library. 

Parameters
     connector_id | IN: Identifier of the connector.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


