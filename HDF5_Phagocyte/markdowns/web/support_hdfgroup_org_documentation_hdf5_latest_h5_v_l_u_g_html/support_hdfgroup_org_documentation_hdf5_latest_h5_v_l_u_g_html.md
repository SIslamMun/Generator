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
  * [ The HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title1 "H2")
  * [ The VOL Abstraction Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title2 "H2")
  * [ VOL Connectors](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title3 "H2")
  * [ Quickstart](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title4 "H2")
  * [ Connector Use](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title5 "H2")
  * [ Adapting HDF5 Software to Use the VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title6 "H2")
  * [ Language Wrappers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title7 "H2")
  * [ Using VOL Connectors With The HDF5 Command-Line Tools](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title8 "H2")
  * [ Compatibility](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Virtual Object Layer (VOL)
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Virtual Object Layer (VOL)
##  Introduction
The virtual object layer is an abstraction layer in the HDF5 library that intercepts all API calls that could potentially access objects in an HDF5 container and forwards those calls to a VOL connector, which implements the storage. The user or application gets the benefit of using the familiar and widely-used HDF5 data model and API, but can map the physical storage of the HDF5 file and objects to storage that better meets the application's data needs.
##  The VOL Abstraction Layer
The VOL lies just under the public API. When a storage-oriented public APIcall is made, the library performs a few sanity checks on the input parameters and then immediately invokes a VOL callback, which resolves to an implementation in the VOL connector that was selected when opening or creating the file. The VOL connector then performs whatever operations are needed before control returns to the library, where any final library operations such as assigning IDs for newly created/opened datasets are performed before returning. This means that, for calls that utilize the VOL, all of the functionality is deferred to the VOL connector and the HDF5 library does very little work. An important consideration of this is that most of the HDF5 caching layers (metadata and chunk caches, page buffering, etc.) will not be available as those are implemented in the HDF5 native VOL connector and cannot be easily reused by external connectors.
![](https://support.hdfgroup.org/documentation/hdf5/latest/vol_architecture.png) The VOL Architecture  
---  
Not all public HDF5 API calls pass through the VOL. Only calls which require manipulating storage go through the VOL and require a VOL connector author to implement the appropriate callback. Dataspace, property list, error stack, etc. calls have nothing to do with storage manipulation or querying and do not use the VOL. This may be confusing when it comes to property list calls, since many of those calls set properties for storage. Property lists are just collections of key-value pairs, though, so a particular VOL connector is not required to set or get properties.
Another thing to keep in mind is that not every VOL connector will implement the full HDF5 public API. In some cases, a particular feature like variable-length types may not have been developed yet or may not have an equivalent in the target storage system. Also, many HDF5 public API calls are specific to the native HDF5 file format and are unlikely to have any use in other VOL connectors. A feature/capabilities flag scheme is being developed to help navigate this.
For more information about which calls go through the VOL and the mechanism by which this is implemented, see the connector author and library internals documentation.
##  VOL Connectors
A VOL connector can be implemented in several ways: 
  * as a shared or static library linked to an application 
  * as a dynamically loaded plugin, implemented as a shared library 
  * and even as an internal connector, built into the HDF5 library itself


This section mostly focuses on external connectors, both libraries and plugins, as those are expected to be much more common than internal implementations.
A list of VOL connectors can be found here: [Registered VOL Connectors](https://support.hdfgroup.org/releases/hdf5/documentation/registered_vol_connectors.md)
This list is incomplete and only includes the VOL connectors that have been registered with The HDF Group.
Not every connector in this collection is actively maintained by The HDF Group. It simply serves as a single location where important VOL connectors can be found. See the documentation in a connector's repository to determine its development status and the parties responsible for it.
A VOL template that contains build scripts (Autotools and CMake) and an empty VOL connector "shell" which can be copied and used as a starting point for building new connectors is located here: [VOL Connector Template](https://github.com/HDFGroup/vol-template)
This template VOL connector is for use in constructing terminal VOL connectors that do not forward calls to an underlying connector. The external pass-through VOL connector listed on the registered connector page can be used as a starting point for pass-through connectors.
The only current (non-test) internal VOL connector distributed with the library is the native file format connector (the "native VOL connector") which contains the code that handles native HDF5 (*.h5/hdf5) files. In other words, even the canonical HDF5 file format is implemented via the VOL, making it a core part of the HDF5 library and not an optional component which could be disabled.
It has not been completely abstracted from the HDF5 library, though, and is treated as a special case. For example, it cannot be unloaded and is always present.
##  Quickstart
The following steps summarize how one would go about using a VOL connector with an application. More information on particular steps can be found later on in this document.
###  Read The Documentation For The New VOL Connector
Many VOL connectors will require specific setup and configuration of both the application and the storage. Specific permissions may have to be set, configuration files constructed, and connector-specific setup calls may need to be invoked in the application. In many cases, converting software to use a new VOL connector will be more than just a straightforward drop-in replacement done by specifying a name in the VOL plugin environment variable.
###  Use A VOL-Enabled HDF5 Library
The virtual object layer was introduced in HDF5 1.12.0, however that version of the VOL is deprecated due to inadequate support for pass-through connectors. These deficiencies have been addressed in HDF5 1.14.0, so VOL users and connector authors should target the 1.14.0 VOL API.
On Windows, it's probably best to use the same debug vs release configuration for the application and all libraries in order to avoid C runtime (CRT) issues. Pre-2015 versions of Visual Studio are not supported.
###  Determine How You Will Set The VOL Connector
Fundamentally, setting a VOL connector involves modifying the file access property list (fapl) that will be used to open or create the file.
There are essentially three ways to do this: 
  * Direct use of [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0)
  * Library-specific API calls that call [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0) for you 
  * Use the VOL environment variable, which will also call [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0) for you


Exactly how you go about setting a VOL connector in a fapl, will depend on the complexity of the VOL connector and how much control you have over the application's source code. Note that the environment variable method, though convenient, has some limitations in its implementation, which are discussed below.
###  If Needed: Update Your Code To Load And Use A VOL Connector
There are two concerns when modifying the application: 
  * It may be convenient to add connector-specific setup calls to the application. 
  * You will also need to protect any API calls which are only implemented in the native VOL connector as those calls will fail when using a non-native VOL connector. See the section [Adapting HDF5 Software to Use the VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#subsec_vol_adapt), below. A list of native VOL API calls has been included in [List of HDF5 Native VOL API Calls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#subsubsec_vol_compat_native).


In some cases, using the VOL environment variable will work well for setting the connector and any associated storage setup and the application will not use API calls that are not supported by the VOL connector. In this case, no application modification will be necessary.
###  If Using A Plugin: Make Sure The VOL Connector Is In The Search
Path The default location for all HDF5 plugins is set at configure time when building the HDF5 library. This is true for both CMake and the Autotools. The default locations for the plugins on both Windows and POSIX systems is listed further on in this document.
###  Optional: Set The VOL Connector Via The Environment Variable
In place of modifying the source code of your application, you may be able to simply set the [HDF5_VOL_CONNECTOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8d21b0c065a27e1b6ea46efb966c7394) environment variable (see below). This will automatically use the specified VOL in place of the native VOL connector.
##  Connector Use
Before a VOL connector can be set in a fapl, it must be registered with the library ([H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0) requires the connector's [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ID) and, if a plugin, it must be discoverable by the library at run time.
###  Registration
Before a connector can be used, it must be registered. This loads the connector into the library and give it an HDF5 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ID. The [H5VLregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b) API calls are used for this. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1)(const char *connector_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8)([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c)
int H5VL_class_value_t
VOL connector identifiers.
**Definition** H5VLpublic.h:175
[H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8)
hid_t H5VLregister_connector_by_value(H5VL_class_value_t connector_value, hid_t vipl_id)
Registers a new VOL connector by value.
[H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1)
hid_t H5VLregister_connector_by_name(const char *connector_name, hid_t vipl_id)
Registers a new VOL connector by name.
When used with a plugin, these functions will check to see if an appropriate plugin with a matching name, value, etc. is already loaded and check the plugin path (see above) for matching plugins if this is not true. The functions return [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) if they are unable to register the connector. Many VOL connectors will provide a connector-specific init call that will load and register the connector for you.
Note the two ways that a VOL connector can be identified: by a name or by a connector-specific numerical value ([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c "VOL connector identifiers.") is typedef’d to an integer). The name and value for a connector can be found in the connector's documentation or public header file.
Each call also takes a VOL initialization property list (vipl). The library adds no properties to this list, so it is entirely for use by connector authors. Set this to [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) unless instructed differently by the documentation for the VOL connector.
As far as the library is concerned, connectors do not need to be explicitly unregistered as the library will unload the plugin and close the ID when the library is closed. If you want to close a VOL connector ID, either [H5VLunregister_connector()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e) or [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094) can be used (they have the same internal code path). The library maintains a reference count on all open IDs and will not do the actual work of closing an ID until its reference count drops to zero, so it's safe to close IDs anytime after they are used, even while an HDF5 file that was opened with that connector is still open.
Note that it's considered an error to unload the native VOL connector. The library will prevent this. This means that, for the time being, the native VOL connector will always be available. This may change in the future so that the memory footprint of the native VOL connector goes away when not in use.
###  Connector Versioning
The VOL connector struct provides a **conn_version** field for versioning connectors. The library developers are working on some best practices for versioning connectors.
###  Connector-Specific Registration Calls
Most connectors will provide a special API call which will set the connector in the fapl. These will often be in the form of **H5Pset_fapl_ <name>()**. For example, the [DAOS VOL](https://github.com/HDFGroup/vol-daos) connector provides a **H5Pset_fapl_daos()** API call which will take MPI parameters and make this call. See the connector's documentation or public header file(s) for more information.
###  H5Pset_vol()
The is the main library API call for setting the VOL connector in a file access property list. Its signature is: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_vol_id, const void new_vol_info)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0)
herr_t H5Pset_vol(hid_t plist_id, hid_t new_vol_id, const void *new_vol_info)
Set the file VOL connector for a file access property list.
It takes the ID of the file access property list, the ID of the registered VOL connector, and a pointer to whatever connector-specific data the connector is expecting. This will usually be a data struct specified in the connector's header or a NULL pointer if the connecter requires no special information (as in the native VOL connector).
As mentioned above, many connectors will provide their own replacement for this call. See the connector's documentation for more information.
###  VOL Connector Search Path
Dynamically loaded VOL connector plugins are discovered and loaded by the library using the same mechanism as dataset/group filter plugins. The default locations are:
_Default locations_
POSIX systems: /usr/local/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/lib/plugin
Windows: %ALLUSERSPROFILE%/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/lib/plugin
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26
These default locations can be overridden by setting the [HDF5_PLUGIN_PATH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a020d112b3c3251e7518461466777611d) environment variable. There are also public H5PL API calls which can be used to add, modify, and remove search paths. The library will only look for plugins in the specified plugin paths. By default, it will NOT find plugins that are simply located in the same directory as the executable.
###  Parameter Strings
Each VOL connector is allowed to take in a parameter string which can be parsed via [H5VLconnector_str_to_info()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f) to get an info struct which can be passed to [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0). 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLconnector_str_to_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f)(const char *str, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void **info)
[H5VLconnector_str_to_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f)
herr_t H5VLconnector_str_to_info(const char *str, hid_t connector_id, void **info)
And the obtained info can be freed via: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfree_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d7c204a3db83d5563b0be557a3a4571)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void *vol_info)
[H5VLfree_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d7c204a3db83d5563b0be557a3a4571)
herr_t H5VLfree_connector_info(hid_t connector_id, void *vol_info)
Most users will not need this functionality as they will be using either connector- specific setup calls which will handle registering and configuring the connector for them or they will be using the environment variable (see below).
###  Environment Variable
The HDF5 library allows specifying a default VOL connector via an environment variable: [HDF5_VOL_CONNECTOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8d21b0c065a27e1b6ea46efb966c7394). The value of this environment variable should be set to ” _vol connector name <parameters>_”.
This will perform the equivalent of: 
  1. [H5VLregister_connector_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1) using the specified connector name 
  2. [H5VLconnector_str_to_info()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f) using the specified parameters. This will go through the connector we got from the previous step and should return a VOL info struct from the parameter string in the environment variable. 
  3. [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0) on the default fapl using the obtained ID and info. 


The environment variable is parsed once, at library startup. Since the environment variable scheme just changes the default connector, it can be overridden by subsequent calls to [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0). The _< parameters>_ is optional, so for connectors which do not require any special configuration parameters you can just set the environment variable to the name.
NOTE: Implementing the environment variable in this way means that setting the native VOL connector becomes somewhat awkward as there is no explicit HDF5 API call to do this. Instead you will need to get the native VOL connector's ID via [H5VLget_connector_id_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee)([H5_VOL_NATIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html#a5768c946c69994f8269b509838a87d89)) and set it manually in the fapl using [H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0).
##  Adapting HDF5 Software to Use the VOL
The VOL was engineered to be as unobtrusive as possible and, when a connector which implements most/all of the data model functionality is in use, many applications will require little, if any, modification. As mentioned in the quick start section, most modifications will probably consist of connector setup code (which can usually be accomplished via the environment variable), adapting code to use the new token-based API calls, and protecting native-VOL-connector-specific functions.
###  haddr_t → H5O_token_t
Some HDF5 API calls and data structures refer to addresses in the HDF5 using the [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) type. Unfortunately, the concept of an ”address” will make no sense for many connectors, though they may still have some sort of location key (e.g.: a key in a key-value pair store).
As a part of the VOL work, the HDF5 API was updated to replace the [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) type with a new [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) type that represents a more generic object location. These tokens appear as an opaque byte array of [H5O_MAX_TOKEN_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#ac91e46b83ee173747f9792b33755ff0e) bytes that is only meaningful for a particular VOL connector. They are not intended for interpretation outside of a VOL connector, though a connector author may provide an API call to convert their tokens to something meaningful for the storage. 
typedef struct [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) {
uint8_t [__data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html#adaac031833a234d10c1ff3130f6aa4cc)[[H5O_MAX_TOKEN_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#ac91e46b83ee173747f9792b33755ff0e)];
} [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html);
[H5O_MAX_TOKEN_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#ac91e46b83ee173747f9792b33755ff0e)
#define H5O_MAX_TOKEN_SIZE
**Definition** H5public.h:428
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)
**Definition** H5public.h:437
[H5O_token_t::__data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html#adaac031833a234d10c1ff3130f6aa4cc)
uint8_t __data[(16)]
**Definition** H5public.h:438
As an example, in the native VOL connector, the token stores an [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address and addresses can be converted to and from tokens using [H5VLnative_addr_to_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga09ca3912386a8c8c66edbcbbe2c10c1f "Convert a haddr_t address to a native VOL connector token.") and [H5VLnative_token_to_addr()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga7136f48f79f4b88d87002d5c218ceb40 "Convert a native VOL connector token to a haddr_t address.").
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLnative_addr_to_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga09ca3912386a8c8c66edbcbbe2c10c1f)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLnative_token_to_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga7136f48f79f4b88d87002d5c218ceb40)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *addr)
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
[H5VLnative_addr_to_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga09ca3912386a8c8c66edbcbbe2c10c1f)
herr_t H5VLnative_addr_to_token(hid_t loc_id, haddr_t addr, H5O_token_t *token)
Convert a haddr_t address to a native VOL connector token.
[H5VLnative_token_to_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga7136f48f79f4b88d87002d5c218ceb40)
herr_t H5VLnative_token_to_addr(hid_t loc_id, H5O_token_t token, haddr_t *addr)
Convert a native VOL connector token to a haddr_t address.
Several API calls have also been added to compare tokens and convert tokens to and from strings.
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Otoken_cmp](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb8da4fbe62f8a3cd9146a7ac1093562)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token1, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token2,
int *cmp_value)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Otoken_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2bdd7528090f7f2c4b361ab4cc7735f6)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token, char **token_str)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Otoken_from_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5136c14b4e907f15007030d7a6d6cd24)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *token_str, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token)
[H5Otoken_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2bdd7528090f7f2c4b361ab4cc7735f6)
herr_t H5Otoken_to_str(hid_t loc_id, const H5O_token_t *token, char **token_str)
Serializes a connector's object token into a string.
[H5Otoken_from_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5136c14b4e907f15007030d7a6d6cd24)
herr_t H5Otoken_from_str(hid_t loc_id, const char *token_str, H5O_token_t *token)
Deserializes a string into a connector object token.
[H5Otoken_cmp](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb8da4fbe62f8a3cd9146a7ac1093562)
herr_t H5Otoken_cmp(hid_t loc_id, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value)
Compares two VOL connector object tokens.
###  Specific API Call Substitutions
####  [H5Fis_hdf5()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225 "Determines whether a file is in the HDF5 format.") → [H5Fis_accessible()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442 "Checks if a file can be opened with a given file access property list.")
[H5Fis_hdf5()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225) does not take a file access property list (fapl). As this is where the VOL connector is specified, this call cannot be used with arbitrary connectors. As a VOL-enabled replacement, [H5Fis_accessible()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442) has been added to the library. It has the same semantics as [H5Fis_hdf5()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225), but takes a fapl so it can work with any VOL connector.
Note that, at this time, [H5Fis_hdf5()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225) always uses the native VOL connector, regardless of the settings of environment variables, etc. 
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5Fis_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442)(const char *container_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)
int htri_t
**Definition** H5public.h:282
[H5Fis_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442)
htri_t H5Fis_accessible(const char *container_name, hid_t fapl_id)
Checks if a file can be opened with a given file access property list.
Note also that an error is only returned if it is not possible to determine if a file (or container) is accessible. It is not an error to return 'false' for whether the file is accessible.
#### H5Oget_info[1|2]() → [H5Oget_info3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013 "Retrieves the metadata for an object specified by an identifier.") and [H5Oget_native_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46 "Retrieve native file format information about an object.")
The [H5Oget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80) and [H5Oget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga06f896e14fe4fa940fbc2bc235e0cf74) family of HDF5 API calls are often used by user code to obtain information about an object in the file, however these calls returned a struct which contained native information and are thus unsuitable for use with arbitrary VOL connectors.
A new [H5Oget_info3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013) family of API calls has been added to the library which only return data model information via a new [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) struct. This struct also returns [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) tokens in place of [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addresses. 
[H5Oget_info3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo, unsigned fields)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_info_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gabb69c962999e027cef0079bbb1282199)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo,
unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_info_by_idx3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa2f8884f7d3e7fd9b8549f5b59fd9eb)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo,
unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9)
H5_iter_order_t
**Definition** H5public.h:379
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3)
H5_index_t
**Definition** H5public.h:402
[H5Oget_info_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gabb69c962999e027cef0079bbb1282199)
herr_t H5Oget_info_by_name3(hid_t loc_id, const char *name, H5O_info2_t *oinfo, unsigned fields, hid_t lapl_id)
Retrieves the metadata for an object, identifying the object by location and relative name.
[H5Oget_info3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013)
herr_t H5Oget_info3(hid_t loc_id, H5O_info2_t *oinfo, unsigned fields)
Retrieves the metadata for an object specified by an identifier.
[H5Oget_info_by_idx3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa2f8884f7d3e7fd9b8549f5b59fd9eb)
herr_t H5Oget_info_by_idx3(hid_t loc_id, const char *group_name, H5_index_t idx_type, H5_iter_order_t order, hsize_t n, H5O_info2_t *oinfo, unsigned fields, hid_t lapl_id)
Retrieves the metadata for an object, identifying the object by an index position.
[H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html)
**Definition** H5Opublic.h:145
typedef struct [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) {
unsigned long [fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aa056ca8d2414b80b34a00c06cd69d29e); // File number that object is located in
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) [token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888); // Token representing the object
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a6e12ef8203e400ee995a41a24e981539); // Basic object type (group, dataset, etc.)
unsigned [rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a299ecaad7f9548089654d47a1b06291f); // Reference count of object
time_t [atime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a8688b65a745fcb4ef567a480b22a65b5); // Access time
time_t [mtime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a4a2d3a53446ef3217564ced73636f9bf); // Modification time
time_t [ctime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a2b105009c2d9020cbaf51d4457c20e3d); // Change time
time_t [btime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#acdd8ce155a94e9a2748566d20ec1f92f); // Birth time
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#abc37a3659a46ce6096446cfd0d9f67ff); // # of attributes attached to object
} [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html);
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec)
H5O_type_t
**Definition** H5Opublic.h:107
[H5O_info2_t::rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a299ecaad7f9548089654d47a1b06291f)
unsigned rc
**Definition** H5Opublic.h:150
[H5O_info2_t::ctime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a2b105009c2d9020cbaf51d4457c20e3d)
time_t ctime
**Definition** H5Opublic.h:153
[H5O_info2_t::mtime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a4a2d3a53446ef3217564ced73636f9bf)
time_t mtime
**Definition** H5Opublic.h:152
[H5O_info2_t::type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a6e12ef8203e400ee995a41a24e981539)
H5O_type_t type
**Definition** H5Opublic.h:149
[H5O_info2_t::atime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a8688b65a745fcb4ef567a480b22a65b5)
time_t atime
**Definition** H5Opublic.h:151
[H5O_info2_t::fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aa056ca8d2414b80b34a00c06cd69d29e)
unsigned long fileno
**Definition** H5Opublic.h:147
[H5O_info2_t::num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#abc37a3659a46ce6096446cfd0d9f67ff)
hsize_t num_attrs
**Definition** H5Opublic.h:155
[H5O_info2_t::btime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#acdd8ce155a94e9a2748566d20ec1f92f)
time_t btime
**Definition** H5Opublic.h:154
[H5O_info2_t::token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888)
H5O_token_t token
**Definition** H5Opublic.h:148
To return the native file format information, [H5Oget_native_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46) calls have been added which can return such data separate from the data model data. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_native_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html) *oinfo, unsigned fields)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_native_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html) *oinfo,
unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_native_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa6570d8b0ef6e2aff75093e1f99f67e)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html) *oinfo,
unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[H5Oget_native_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476)
herr_t H5Oget_native_info_by_name(hid_t loc_id, const char *name, H5O_native_info_t *oinfo, unsigned fields, hid_t lapl_id)
Retrieve native file format information about an object given its name.
[H5Oget_native_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46)
herr_t H5Oget_native_info(hid_t loc_id, H5O_native_info_t *oinfo, unsigned fields)
Retrieve native file format information about an object.
[H5Oget_native_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa6570d8b0ef6e2aff75093e1f99f67e)
herr_t H5Oget_native_info_by_idx(hid_t loc_id, const char *group_name, H5_index_t idx_type, H5_iter_order_t order, hsize_t n, H5O_native_info_t *oinfo, unsigned fields, hid_t lapl_id)
Retrieve native file format information about an object according to the order of an index.
[H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html)
**Definition** H5Opublic.h:164
typedef struct [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html) {
[H5O_hdr_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html) [hdr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a6cfeed1003ece5686449118f36dce19f); // Object header information
// Extra metadata storage for obj & attributes
struct {
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) [obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#ad99b3bb65fd4d3f3ec75ab5eea43eb15); // v1/v2 B-tree & local/fractal heap for groups,
// B-tree for chunked datasets
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) [attr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#aba95c19d6477004a0fa4591e467412b6); // v2 B-tree & heap for attributes
} [meta_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a73a8b12326b16b529bf531d6023ad8b8);
} [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html);
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html)
**Definition** H5public.h:414
[H5O_hdr_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html)
**Definition** H5Opublic.h:122
[H5O_native_info_t::hdr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a6cfeed1003ece5686449118f36dce19f)
H5O_hdr_info_t hdr
**Definition** H5Opublic.h:165
[H5O_native_info_t::meta_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a73a8b12326b16b529bf531d6023ad8b8)
struct H5O_native_info_t::@260162027145300125277002203123231341201164141000 meta_size
[H5O_native_info_t::attr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#aba95c19d6477004a0fa4591e467412b6)
H5_ih_info_t attr
**Definition** H5Opublic.h:168
[H5O_native_info_t::obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#ad99b3bb65fd4d3f3ec75ab5eea43eb15)
H5_ih_info_t obj
**Definition** H5Opublic.h:167
#### H5Ovisit[1|2]() → [H5Ovisit3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a "Recursively visits all objects accessible from a specified object.")
The callback used in the [H5Ovisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) family of API calls took an H5O info t struct parameter. As in [H5Oget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5), this both commingled data model and native file format information and also used native HDF5 file addresses.
New [H5Ovisit3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a) API calls have been created which use the token-based, data-model-only [H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4) struct in the callback.
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Ovisit3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c) op,
void *op_data, unsigned fields)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Ovisit_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga34815400b01df59c4dac19436124885a)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c) op, void *op_data,
unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c)
herr_t(* H5O_iterate2_t)(hid_t obj, const char *name, const H5O_info2_t *info, void *op_data)
**Definition** H5Opublic.h:195
[H5Ovisit_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga34815400b01df59c4dac19436124885a)
herr_t H5Ovisit_by_name3(hid_t loc_id, const char *obj_name, H5_index_t idx_type, H5_iter_order_t order, H5O_iterate2_t op, void *op_data, unsigned fields, hid_t lapl_id)
Recursively visits all objects accessible from a specified object.
[H5Ovisit3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a)
herr_t H5Ovisit3(hid_t obj_id, H5_index_t idx_type, H5_iter_order_t order, H5O_iterate2_t op, void *op_data, unsigned fields)
Recursively visits all objects accessible from a specified object.
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj, const char *name, const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *info, void *op_data)
####  [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) → [H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.")
The [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) API calls were updated to use tokens instead of addresses in the [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) struct. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)
herr_t H5Lget_info2(hid_t loc_id, const char *name, H5L_info2_t *linfo, hid_t lapl_id)
Returns information about a link.
[H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)
herr_t H5Lget_info_by_idx2(hid_t loc_id, const char *group_name, H5_index_t idx_type, H5_iter_order_t order, hsize_t n, H5L_info2_t *linfo, hid_t lapl_id)
Retrieves metadata for a link in a group, according to the order within a field or index.
[H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)
Information struct for links.
**Definition** H5Lpublic.h:76
typedef struct {
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) type; // Type of link
bool corder_valid; // Indicate if creation order is valid
int64_t corder; // Creation order
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) cset; // Character set of link name
union {
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token; // Token of location that hard link points to
size_t val_size; // Size of a soft link or UD link value
} u;
} [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html);
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48)
H5L_type_t
Link class types.
**Definition** H5Lpublic.h:57
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a)
H5T_cset_t
**Definition** H5Tpublic.h:94
####  [H5Literate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) and [H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) → [H5Literate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index.") and [H5Lvisit2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66 "Recursively visits all links starting from a specified group.")
The callback used in these API calls used the old [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) struct, which used addresses instead of tokens. These callbacks were versioned in the C library and now take modified [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9 "Prototype for H5Literate2\(\), H5Literate_by_name2\(\) operator.") callbacks which use the new token-based info structs. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Literate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx,
[H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Literate_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op,
void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Lvisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op,
void *op_data)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Lvisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)
[H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)
herr_t(* H5L_iterate2_t)(hid_t group, const char *name, const H5L_info2_t *info, void *op_data)
Prototype for H5Literate2(), H5Literate_by_name2() operator.
**Definition** H5Lpublic.h:94
[H5Literate_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e)
herr_t H5Literate_by_name2(hid_t loc_id, const char *group_name, H5_index_t idx_type, H5_iter_order_t order, hsize_t *idx, H5L_iterate2_t op, void *op_data, hid_t lapl_id)
Iterates through links in a group.
[H5Literate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73)
herr_t H5Literate2(hid_t grp_id, H5_index_t idx_type, H5_iter_order_t order, hsize_t *idx, H5L_iterate2_t op, void *op_data)
Iterates over links in a group, with user callback routine, according to the order within an index.
[H5Lvisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66)
herr_t H5Lvisit2(hid_t grp_id, H5_index_t idx_type, H5_iter_order_t order, H5L_iterate2_t op, void *op_data)
Recursively visits all links starting from a specified group.
[H5Lvisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173)
herr_t H5Lvisit_by_name2(hid_t loc_id, const char *group_name, H5_index_t idx_type, H5_iter_order_t order, H5L_iterate2_t op, void *op_data, hid_t lapl_id)
Recursively visits all links starting from a specified group.
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group, const char *name, const [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *info,
void *op_data);
#### H5Oopen by addr() → H5Oopen by token()
The new [H5Oopen_by_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262) API call can be used to open objects by the tokens that are returned by the various ”get info”, et al. API calls. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Oopen_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token)
[H5Oopen_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262)
hid_t H5Oopen_by_token(hid_t loc_id, H5O_token_t token)
Opens an object in an HDF5 file using its VOL independent token.
###  Protect Native-Only API Calls
In HDF5 1.14.0, a way to determine support for optional calls has been added. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLquery_optional](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, int opt_type, uint64_t *flags)
[H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599)
H5VL_subclass_t
**Definition** H5VLpublic.h:183
[H5VLquery_optional](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043)
herr_t H5VLquery_optional(hid_t obj_id, H5VL_subclass_t subcls, int opt_type, uint64_t *flags)
Determine if a VOL connector supports a particular optional callback operation.
The call takes an object that is VOL managed (i.e.; file, group, dataset, attribute, object, committed datatype), the VOL subclass (an enum documented in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html)), an operation ”type” (discussed below), and an out parameter for the bitwise capabilities flags (also discussed below). Code that needs to protect a VOL-specific API call can call the function to see if the API call is supported, which will be reported via the flags. Specifically, if the [H5VL_OPT_QUERY_SUPPORTED](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga2208d2bf3252e8201b80d48d9bfdd26c) bit is set, the feature is supported. The other flags are more useful for VOL connector authors than end users.
In the case of the native VOL connector, the opt type operations are documented in [H5VLnative.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html). The current list of native operations is given at the end of this document, along with a list of native-only connector calls.
##  Language Wrappers
Due to the parameter type and callback changes that were required in the C library API regarding the update from [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addresses to [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) tokens and the difficulty in versioning the wrapper APIs, it was decided to update all of the wrappers to use tokens instead of addresses. This will allow the language wrappers to make use of the VOL, but at the expense of backward compatibility.
Information on the C API changes can be found above.
Affected API calls, by language:
###  C++
  * The **visit_operator_t** callback now uses a [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) parameter instead of [H5O_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info1__t.html) so the callback can be passed to [H5Ovisit3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a) internally. This affects the H5Object::visit() method. 
  * The H5Location::getObjinfo() methods now take [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) parameters. 
  * The H5Location::getLinkInfo() methods now return [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html "Information struct for links.") structs. 
  * H5File::isHdf5 uses [H5Fis_accessible()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442), though it always passes [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) as the fapl. It will only work with arbitrary VOL connectors if the default VOL connector is changed via the environment variable. 


The C++ wrappers do not allow opening HDF5 file objects by address or token.
The public H5VL API calls found in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) were NOT added to the C++ API.
###  Fortran
As in the C API, these API calls had their structs updated to the token version so the h5o_info_t, etc. structs no longer contain native file format information and the callbacks will need to match the non-deprecated, token-enabled versions. 
  * [h5lget_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l.html#gaeac9913b7460534637eb563bb24eeb3e)
  * [h5lget_info_by_idx_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l.html#gaf01b68f47bbe14f84531b70b479b9ed3)
  * [h5literate_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l.html#gad003be2bb372d942d6da17786d6f3780)
  * [h5literate_by_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l.html#gaefb80ce5f9684adc388db149145a56a9)
  * [h5oget_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#gabdbe70d333edbc46cffd791495e3edea)
  * [h5oget_info_by_idx_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#ga6666adcfef409c0828390b75730f9987)
  * [h5oget_info_by_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#ga40081a5f47dc7900a795c0df62791ff7)
  * [h5oopen_by_token_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#ga5ce7a8d47ba39d5f2e694a67aec1b977)
  * [h5ovisit_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#ga1aa4f84b78f029f048593b1ec0757a63)
  * [h5ovisit_by_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html#gaed6f1ee04db6973cbffca2cf0c33348f)


Additionally, [h5fis_hdf5_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_f.html#gad4e7f5e1a34daabcdc5a16a4df25f317) was updated to use [H5Fis_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442) internally, though with the same caveat as the C++ implementation: the default fapl is always passed in so arbitrary VOL connectors will only work if the default VOL connector is changed via the environment variable.
The public H5VL API calls found in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) were also added to the Fortran wrappers.
###  Java/JNI
  * [H5Fis_hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225) Will fail when the library is built without deprecated symbols. 
  * [H5Fis_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442) is available and takes a fapl, allowing it to work with arbitrary VOL connectors. 
  * The [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)(O|L)get_info, [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)(O|L)visit, and [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) calls were updated as in the C library. 
  * [H5Oget_native_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476) et al. were added and they work as in the C library (e.g.: essentially native VOL connector only). 
  * [H5Oopen_by_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga137f3823adab4daaaf8fe87b40453fa2) was replaced with [H5Oopen_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262). 
  * The public API calls in [H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) were added to the JNI. 


##  Using VOL Connectors With The HDF5 Command-Line Tools
The following command-line tools are VOL-aware and can be used with arbitrary VOL connectors: 
  * (p)[h5diff](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_f__u_g.html#sec_cltools_h5diff)
  * [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)
  * [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)
  * [h5mkgrp](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__m_g__u_g.html#sec_cltools_h5mkgrp)
  * [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack)


The VOL connector can be set either using the [HDF5_VOL_CONNECTOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8d21b0c065a27e1b6ea46efb966c7394) environment variable (see above) or via the command line. Each of the above tools takes command-line options to set the VOL connector by name or value and the VOL connector string, usually in the form of 
--vol-(name|value|info)
See the individual tool's help for the options specific to that tool.
##  Compatibility
###  List of HDF5 Native VOL API Calls
These API calls will probably fail when used with terminal VOL connectors other than the native HDF5 file format connector. Their use should be protected in code that uses arbitrary VOL connectors. Note that some connectors may, in fact, implement some of this functionality as it is possible to mimic the native HDF5 connector, however this will probably not be true for most non-native VOL connectors. 
Alphabetical list of HDF5 API calls specific to the native VOL connector API  | Description   
---|---  
[H5Aget_num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaadd809fc16238754105bbddd20bcdde1 "Determines the number of attributes attached to an object.") | deprecated   
[H5Aiterate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.") | deprecated   
[H5Dchunk_iter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac482c2386aa3aea4c44730a627a7adb8 "Iterate over all chunks of a chunked dataset.") |   
H5Ddebug  | Internal API routines   
H5Dformat_convert  | Internal API routines   
H5Dget_chunk_index_type  | Internal API routines   
[H5Dget_chunk_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaccff213d3e0765b86f66d08dd9959807 "Retrieves information about a chunk specified by its index.") |   
[H5Dget_chunk_info_by_coord](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0 "Retrieves information about a chunk specified by its coordinates.") |   
[H5Dget_chunk_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234 "Returns the amount of storage allocated within the file for a raw data chunk in a dataset.") |   
[H5Dget_num_chunks](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8e15897dcc5799d6c09806644b492d7a "Retrieves number of chunks that have nonempty intersection with a specified selection.") |   
[H5Dget_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3 "Returns dataset address in file.") |   
[H5Dread_chunk1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.") |   
[H5Dread_chunk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") |   
[H5Dwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") |   
H5FD*  |   
[H5Fclear_elink_file_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gafcc153d8606829d4401e93305e5246d7 "Clears the external link open file cache.") |   
H5Fformat_convert  | Internal API routines   
[H5Fget_dset_no_attrs_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae688a27695349925f08eaa3c44f99265 "Retrieves the setting for whether or not a file will create minimized dataset object headers.") |   
[H5Fget_eoa](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga4c18bddafc652203944d889a602bd53f "Retrieves the file's end-of-allocation \(EOA\)") |   
[H5Fget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file.") |   
[H5Fget_filesize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga515426821321c261a825b4e4a3f576fe "Returns the size of an HDF5 file \(in bytes\)") |   
[H5Fget_free_sections](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gab9cbf1a45f9dcda34b43f985b7848434 "Retrieves free-space section information for a file.") |   
[H5Fget_freespace](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3ef2673183567543346668a8f1eca2e9 "Returns the amount of free space in a file \(in bytes\)") |   
[H5Fget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga660153029322fa6b77f5473cedc2d72f "Retrieves global file information.") | deprecated   
[H5Fget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e "Retrieves global file information.") |   
[H5Fget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59 "Obtains current metadata cache configuration for target file.") |   
[H5Fget_mdc_hit_rate](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f "Obtains target file's metadata cache hit rate.") |   
[H5Fget_mdc_image_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a "Obtains information about a cache image if it exists.") |   
[H5Fget_mdc_logging_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") |   
[H5Fget_mdc_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f "Obtains current metadata cache size data for specified file.") |   
[H5Fget_metadata_read_retry_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#gaa80bd62f19993e414e383db7d1623e5f "Retrieves the collection of read retries for metadata entries with checksum.") |   
[H5Fget_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd "Retrieves the atomicity mode in use.") |   
[H5Fget_page_buffering_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga0663defe0143631f4292267c21e94202 "Retrieves statistics about page access when it is enabled.") |   
[H5Fget_vfd_handle](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") |   
[H5Fincrement_filesize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadbe82c1f6e16c21062fabd20b0ffccd4 "Sets the file' EOA to the maximum of \(EOA, EOF\) + increment.") |   
[H5Fis_hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225 "Determines whether a file is in the HDF5 format.") | deprecated   
[H5Freset_mdc_hit_rate_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f "Resets hit rate statistics counters for the target file.") |   
[H5Freset_page_buffering_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga7ef1c0aab9a7a9112a8d0a788ec8696c "Resets the page buffer statistics.") |   
[H5Fset_dset_no_attrs_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaddd372973b35cde53300817d3ffaee5a "Sets the flag to create minimized dataset object headers.") |   
[H5Fset_latest_format](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa2a6549a4b661e70ededfabea6faf0fa "Sets the latest version of the library to be used for writing objects.") | deprecated   
[H5Fset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga4b833c33fe2e141a26b6f2ad559d3610 "Enables the switch of version bounds setting for a file.") |   
[H5Fset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8 "Attempts to configure metadata cache of target file.") |   
[H5Fset_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") |   
[H5Fstart_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") |   
[H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe "Retrieves free-space section information for a file.") |   
[H5Fstop_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") |   
[H5Gget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gac5a1c3e1ed4264d92cc02ca20afc57f4 "Retrieves comment for specified object.") | deprecated   
[H5Giterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga957fee64f796f184f542537127ad6c11 "Iterates over the entries of a group invoking a callback for each entry encountered.") | deprecated   
[H5Gget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad4be126ab7bbf2001435e8e70089f3d3 "Retrieves information about a group.") |   
[H5Gget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadedd0c73c98f2ada69305f2992c3300e "Retrieves information about a group by its name.") |   
[H5Gget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga985f27ad1a164d99fa1f58c6de60ab00 "Retrieves information about a group, according to the group's position within an index.") |   
[H5Gget_objinfo](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga471d40a03e09a9d7fe0c7c7c31a453fa "Returns information about an object.") | deprecated   
[H5Gget_objname_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga80180e7b819d3c9b3b3f1895e9baaf5b "Returns the name of an object specified by an index.") | deprecated   
[H5Gget_objtype_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gab1383b8cca3fa99410ad36427059c5a7 "Returns the type of an object specified by an index.") | deprecated   
[H5Gset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga7a615715ea68fc1bf11484a8278fe682 "Sets comment for specified object.") | deprecated   
[H5Lget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.") | deprecated   
[H5Lget_info_by_idx1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a "Retrieves metadata for a link in a group, according to the order within a field or index.") | deprecated   
[H5Literate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1 "Iterates over links in a group, with user callback routine, according to the order within an index.") | deprecated   
[H5Literate_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga87e036da0c8d1146a073f3ee08e0fedc "Iterates through links in a group by its name.") | deprecated   
[H5Lvisit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga5424ef7043c82147490d027a0e8a59ef "Recursively visits all links starting from a specified group.") | deprecated   
[H5Lvisit_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1f1ba1bb4d44f2c111990024809417ac "Recursively visits all links starting from a specified group.") | deprecated   
[H5Oare_mdc_flushes_disabled](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gac3b3b768a1c50be46e63cf5a631b30e3 "Retrieves comment for specified object.") |   
[H5Odisable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga0908be309da1fb4f771c1e264fac22ae "Prevents metadata entries for an HDF5 object from being flushed from the metadata cache to storage.") |   
[H5Oenable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga21014920bdabf6973e233796d7174156 "Enables flushing of dirty metadata entries from a file's metadata cache.") |   
[H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b "Retrieves comment for specified object.") |   
[H5Oget_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gae6d92d597c5a292d342a1bda91e41171 "Retrieves comment for specified object.") |   
[H5Oget_info_by_idx1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga7208d2cf198dcfc875603323841bffae "Retrieves the metadata for an object, identifying the object by an index position.") | deprecated   
[H5Oget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga85e15e65922874111da1a5efd5dd7bed "Retrieves the metadata for an object, identifying the object by an index position.") | deprecated   
[H5Oget_info_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga96ce408ffda805210844246904da2842 "Retrieves the metadata for an object, identifying the object by location and relative name.") | deprecated   
[H5Oget_info_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga0090da86c086c1c63a5acfaed39a035e "Retrieves the metadata for an object, identifying the object by location and relative name.") | deprecated   
[H5Oget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80 "Retrieves the metadata for an object specified by an identifier.") | deprecated   
[H5Oget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga06f896e14fe4fa940fbc2bc235e0cf74 "Retrieves the metadata for an object specified by an identifier.") | deprecated   
[H5Oget_native_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46 "Retrieve native file format information about an object.") |   
[H5Oget_native_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa6570d8b0ef6e2aff75093e1f99f67e "Retrieve native file format information about an object according to the order of an index.") |   
[H5Oget_native_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476 "Retrieve native file format information about an object given its name.") |   
[H5Oopen_by_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga137f3823adab4daaaf8fe87b40453fa2 "Opens an object using its address within an HDF5 file.") | deprecated   
[H5Oset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga8b5cf8e916204e29616516046121f631 "Sets comment for specified object.") |   
[H5Oset_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafeb5242de7f1080b5c19f4fe19784505 "Sets comment for specified object.") |   
[H5Ovisit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6efdb2a0a9fe9fe46695cc0f7bd993e7 "Recursively visits all objects accessible from a specified object.") | deprecated   
[H5Ovisit_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaffacf3bd66f4fe074099eae1c80914f2 "Recursively visits all objects starting from a specified object.") | deprecated   
[H5Ovisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa4ab542f581f4fc9a4eaa95debb29c9e "Recursively visits all objects accessible from a specified object.") | deprecated   
[H5Ovisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9c155caf5499405fe403e1eb27b5beb6 "Recursively visits all objects starting from a specified object.") | deprecated   
###  List of HDF5 VOL-Independent API Calls
These HDF5 API calls do not depend on a particular VOL connector being loaded. 
Alphabetical list of VOL-independent HDF5 API calls API  | Description   
---|---  
H5*  |   
[H5Dfill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8d4a57e2b2b8c95cfecf6f75bdaa8343 "Fills dataspace elements with a fill value in a memory buffer.") |   
[H5Dgather](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be "Gathers data from a selection within a memory buffer raw data chunk in a dataset.") |   
[H5Diterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga71421c684884ab49765748720fe938db "Iterates over all selected elements in a dataspace.") |   
[H5Dscatter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3525b15235ba1fd415f988899e48dc5c "Scatters data into a selection within a memory buffer.") |   
[H5Dvlen_reclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119 "Reclaims variable-length \(VL\) datatype memory buffers.") | deprecated   
[H5Dvlen_get_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76 "Determines the number of bytes required to store variable-length \(VL\) data.") |   
H5E*  |   
H5I*  |   
[H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2 "Determines whether a class of user-defined links is registered.") |   
[H5Lregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class.") |   
[H5Lunpack_elink_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.") |   
[H5Lunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec "Unregisters a class of user-defined links.") |   
H5PL*  |   
H5P*  |   
H5S*  |   
H5T*  | non-committed   
H5VL*  |   
H5Z*  |   
###  List of Native VOL Optional Operation Values By Subclass
These values can be passed to the opt type parameter of H5VLquery optional(). 
List of Native VOL Optional Operation Values By Subclass Subclass  | API Reference  | Definition   
---|---|---  
H5VL_SUBCLS_ATTR  |  [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) (deprecated routine)  |  [H5VL_NATIVE_ATTR_ITERATE_OLD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a6f6beb074a558b1b41c48275f25ad243)  
H5VL_SUBCLS_DATASET  | H5Dformat_convert (internal)  |  [H5VL_NATIVE_DATASET_FORMAT_CONVERT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#acee682362743a2d93822b948820f9c6e)  
H5Dget_chunk_index_type (internal)  |  [H5VL_NATIVE_DATASET_GET_CHUNK_INDEX_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#acd4d7845233f37ee36f5f6077f13d194)  
[H5Dget_chunk_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234 "Returns the amount of storage allocated within the file for a raw data chunk in a dataset.") |  [H5VL_NATIVE_DATASET_GET_CHUNK_STORAGE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#acddd86100b4d589e79581edf4884af78)  
[H5Dget_num_chunks](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8e15897dcc5799d6c09806644b492d7a "Retrieves number of chunks that have nonempty intersection with a specified selection.") |  [H5VL_NATIVE_DATASET_GET_NUM_CHUNKS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a67c5302300de3dce3d6a87fc993941e1)  
[H5Dget_chunk_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaccff213d3e0765b86f66d08dd9959807 "Retrieves information about a chunk specified by its index.") |  [H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ad9e5d19fabe5b8365f577d90322e8a80)  
[H5Dget_chunk_info_by_coord](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0 "Retrieves information about a chunk specified by its coordinates.") |  [H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_COORD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#aaad5b69dec6724988d35c02371e374a0)  
[H5Dread_chunk1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.") / [H5Dread_chunk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") |  [H5VL_NATIVE_DATASET_CHUNK_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ae9f6fea99a00fc76f94a3883d936101c)  
[H5Dwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") |  [H5VL_NATIVE_DATASET_CHUNK_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a0d7bc94d260e3e2df060fd9f661bee83)  
[H5Dvlen_get_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76 "Determines the number of bytes required to store variable-length \(VL\) data.") |  [H5VL_NATIVE_DATASET_GET_VLEN_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ad99a75d4e8d9f11ef99f27156c5c8792)  
[H5Dget_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3 "Returns dataset address in file.") |  [H5VL_NATIVE_DATASET_GET_OFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#aae3ec47581e91f02880c16776324edde)  
[H5Dget_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3 "Returns dataset address in file.") |  [H5VL_NATIVE_DATASET_CHUNK_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ab927ce76e0e66748107392eaefea30ad)  
H5VL_SUBCLS_FILE  |  [H5Fclear_elink_file_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gafcc153d8606829d4401e93305e5246d7 "Clears the external link open file cache.") |  [H5VL_NATIVE_FILE_CLEAR_ELINK_CACHE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ac6d0b06911f656afb9769a86aa67ec68)  
[H5Fget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file.") |  [H5VL_NATIVE_FILE_GET_FILE_IMAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ad8169c70f8043b0e9cffc7ccd6a248e3)  
[H5Fget_free_sections](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gab9cbf1a45f9dcda34b43f985b7848434 "Retrieves free-space section information for a file.") |  [H5VL_NATIVE_FILE_GET_FREE_SECTIONS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a6fe307fec4788a2c9cbeccb41d9d86c7)  
[H5Fget_freespace](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3ef2673183567543346668a8f1eca2e9 "Returns the amount of free space in a file \(in bytes\)") |  [H5VL_NATIVE_FILE_GET_FREE_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a9146c2e04ab8c75a6808927bad8b9505)  
[H5Fget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga660153029322fa6b77f5473cedc2d72f "Retrieves global file information.") / [H5Fget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e "Retrieves global file information.") |  [H5VL_NATIVE_FILE_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ac0ff62fa52184866b231d3d314fba88e)  
[H5Fget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59 "Obtains current metadata cache configuration for target file.") |  [H5VL_NATIVE_FILE_GET_MDC_CONF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#af4c416a95c1c2989493918f37d5796aa)  
[H5Fget_mdc_hit_rate](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f "Obtains target file's metadata cache hit rate.") |  [H5VL_NATIVE_FILE_GET_MDC_HR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a62eeb6aab65f7612ad3ce3febbd57fad)  
[H5Fget_mdc_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f "Obtains current metadata cache size data for specified file.") |  [H5VL_NATIVE_FILE_GET_MDC_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ae0d3e6c976b8a9bae3d66cbaa1d9097b)  
[H5Fget_filesize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga515426821321c261a825b4e4a3f576fe "Returns the size of an HDF5 file \(in bytes\)") |  [H5VL_NATIVE_FILE_GET_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ad549326c304f0db34a175f224c0dab38)  
[H5Fget_vfd_handle](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") |  [H5VL_NATIVE_FILE_GET_VFD_HANDLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a3980f03a7f76e028027362a3369c0fb6)  
[H5Freset_mdc_hit_rate_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f "Resets hit rate statistics counters for the target file.") |  [H5VL_NATIVE_FILE_RESET_MDC_HIT_RATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#acc4faec86feb9c1f0f082b798797d630)  
[H5Fset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8 "Attempts to configure metadata cache of target file.") |  [H5VL_NATIVE_FILE_SET_MDC_CONFIG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a4d758b2a05fdbfbde4ea57bf1115dca8)  
[H5Fget_metadata_read_retry_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#gaa80bd62f19993e414e383db7d1623e5f "Retrieves the collection of read retries for metadata entries with checksum.") |  [H5VL_NATIVE_FILE_GET_METADATA_READ_RETRY_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#af3447d71c7e5893c4274c49eb9a8cbe4)  
[H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe "Retrieves free-space section information for a file.") |  [H5VL_NATIVE_FILE_START_SWMR_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a81665bce380069aa23b257699e768ea6)  
[H5Fstart_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") |  [H5VL_NATIVE_FILE_START_MDC_LOGGING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a204824c1386c338c51889a4683313910)  
[H5Fstop_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") |  [H5VL_NATIVE_FILE_STOP_MDC_LOGGING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a43644b7a555039664e5838917dc8b19f)  
[H5Fget_mdc_logging_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") |  [H5VL_NATIVE_FILE_GET_MDC_LOGGING_STATUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a45ab72336cb9193e0e588685773b525f)  
H5Fformat_convert (internal)  |  [H5VL_NATIVE_FILE_FORMAT_CONVERT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#aa0ee2a329e7eedb415411a266a8c48fd)  
[H5Freset_page_buffering_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga7ef1c0aab9a7a9112a8d0a788ec8696c "Resets the page buffer statistics.") |  [H5VL_NATIVE_FILE_RESET_PAGE_BUFFERING_STATS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#acae57ad1a2d0b3918ec43525d691bc2b)  
[H5Fget_page_buffering_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga0663defe0143631f4292267c21e94202 "Retrieves statistics about page access when it is enabled.") |  [H5VL_NATIVE_FILE_GET_PAGE_BUFFERING_STATS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ae28f5fc045e3f3728e582aa3a66721bd)  
[H5Fget_mdc_image_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a "Obtains information about a cache image if it exists.") |  [H5VL_NATIVE_FILE_GET_MDC_IMAGE_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#afa5b76d65952391b8d302aeb14509b3b)  
[H5Fget_eoa](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga4c18bddafc652203944d889a602bd53f "Retrieves the file's end-of-allocation \(EOA\)") |  [H5VL_NATIVE_FILE_GET_EOA](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#ac5230dd379a463ce78f8eb1ca2ade657)  
[H5Fincrement_filesize](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadbe82c1f6e16c21062fabd20b0ffccd4 "Sets the file' EOA to the maximum of \(EOA, EOF\) + increment.") |  [H5VL_NATIVE_FILE_INCR_FILESIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#aa7c0ecdd8fb3f1725ddc647cc15426f3)  
[H5Fset_latest_format](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa2a6549a4b661e70ededfabea6faf0fa "Sets the latest version of the library to be used for writing objects.")/[H5Fset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga4b833c33fe2e141a26b6f2ad559d3610 "Enables the switch of version bounds setting for a file.") |  [H5VL_NATIVE_FILE_SET_LIBVER_BOUNDS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a434d6fea6ce5575042b6ffec0dff58c7)  
[H5Fget_dset_no_attrs_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae688a27695349925f08eaa3c44f99265 "Retrieves the setting for whether or not a file will create minimized dataset object headers.") |  [H5VL_NATIVE_FILE_GET_MIN_DSET_OHDR_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a08d3e870d99f7098c354640b7346597b)  
[H5Fset_dset_no_attrs_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaddd372973b35cde53300817d3ffaee5a "Sets the flag to create minimized dataset object headers.") |  [H5VL_NATIVE_FILE_SET_MIN_DSET_OHDR_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a9c87d8969f22eb8e3002ceb98b001f90)  
[H5Fget_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#ga5356a1bee667b2a7dd1b657c820755fd "Retrieves the atomicity mode in use.") |  [H5VL_NATIVE_FILE_GET_MPI_ATOMICITY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a006f5bedc78630dc1b39671151823d78)  
[H5Fset_mpi_atomicity](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_h5_f.html#gaaefabc54e36a2d13325e9a853400f359 "Sets the MPI atomicity mode.") |  [H5VL_NATIVE_FILE_SET_MPI_ATOMICITY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#aedfd9c1c6b47a9982c804e66a2c2e9dd)  
Adjust file after open, with wrapping context  |  [H5VL_NATIVE_FILE_POST_OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a087662fec2c046e4ede6a7176209babc)  
H5VL_SUBCLS_GROUP  |  [H5Giterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga957fee64f796f184f542537127ad6c11 "Iterates over the entries of a group invoking a callback for each entry encountered.") (deprecated routine)  |  [H5VL_NATIVE_GROUP_ITERATE_OLD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a6e84a71e8968331693da4bd1ffe02783)  
[H5Gget_objinfo](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga471d40a03e09a9d7fe0c7c7c31a453fa "Returns information about an object.") (deprecated routine)  |  [H5VL_NATIVE_GROUP_GET_OBJINFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a060b202dbbb4ce73a85d94de9d33bcb5)  
H5VL_SUBCLS_OBJECT  |  [H5Gget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gac5a1c3e1ed4264d92cc02ca20afc57f4 "Retrieves comment for specified object."), [H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b "Retrieves comment for specified object."), [H5Oget_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gae6d92d597c5a292d342a1bda91e41171 "Retrieves comment for specified object.") |  [H5VL_NATIVE_OBJECT_GET_COMMENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a606309d1a8734beeffb0fe241b30e1e7)  
[H5Gset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga7a615715ea68fc1bf11484a8278fe682 "Sets comment for specified object."), [H5Oset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga8b5cf8e916204e29616516046121f631 "Sets comment for specified object."), [H5Oset_comment_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafeb5242de7f1080b5c19f4fe19784505 "Sets comment for specified object.") |  [H5VL_NATIVE_OBJECT_SET_COMMENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a8ac55735c5782cc6ba6c78df4e3036e2)  
[H5Odisable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga0908be309da1fb4f771c1e264fac22ae "Prevents metadata entries for an HDF5 object from being flushed from the metadata cache to storage.") |  [H5VL_NATIVE_OBJECT_DISABLE_MDC_FLUSHES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a099b9e6c4e1f46a11ede40e95e58e39e)  
[H5Oenable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga21014920bdabf6973e233796d7174156 "Enables flushing of dirty metadata entries from a file's metadata cache.") |  [H5VL_NATIVE_OBJECT_ENABLE_MDC_FLUSHES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a844314a5498935a7570fc69779a6eff3)  
[H5Oare_mdc_flushes_disabled](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gac3b3b768a1c50be46e63cf5a631b30e3 "Retrieves comment for specified object.") |  [H5VL_NATIVE_OBJECT_ARE_MDC_FLUSHES_DISABLED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#a6a27db65f6b7711c13b46d6e9f98d18b)  
[H5Oget_native_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46 "Retrieve native file format information about an object."), [H5Oget_native_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa6570d8b0ef6e2aff75093e1f99f67e "Retrieve native file format information about an object according to the order of an index."), [H5Oget_native_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476 "Retrieve native file format information about an object given its name.") |  [H5VL_NATIVE_OBJECT_GET_NATIVE_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html#af4a0fd6a4385f0eac1c34df9b7bece80)/td>  
Previous Chapter [Properties and Property Lists in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#sec_plist) - Next Chapter [The HDF5 Event Set Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_s__u_g.html#sec_async)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


