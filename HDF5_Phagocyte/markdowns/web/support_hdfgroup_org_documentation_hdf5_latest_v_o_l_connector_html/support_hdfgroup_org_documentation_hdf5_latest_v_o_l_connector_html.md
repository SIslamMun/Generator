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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title0 "H1")
  * [ Creating a New Connector](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title1 "H1")
  * [ Overview](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title2 "H2")
  * [ The HDF5 1.12.x VOL Interface Is DEPRECATED](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title3 "H2")
  * [ VOL-Related HDF5 Header Files](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title4 "H2")
  * [ Library vs Plugin vs Internal](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title5 "H2")
  * [ Build Files / VOL Template](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title6 "H2")
  * [ H5VL_class_t Boilerplate](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title7 "H2")
  * [ Initialization and Shutdown](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title8 "H2")
  * [ Map Storage to HDF5 File Objects](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title9 "H2")
  * [ Fill In VOL Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title10 "H2")
  * [ Handling Optional Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title11 "H2")
  * [ Testing Your Connector](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title12 "H2")
  * [ Passthrough Connectors](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title13 "H2")
  * [ Asynchronous Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title14 "H2")
  * [ VOL Connector Interface Reference](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title15 "H1")
  * [ Mapping the API to the Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title16 "H2")
  * [ Connector Information Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title17 "H2")
  * [ Object Wrap Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title18 "H2")
  * [ The Attribute Function Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title19 "H2")
  * [ Dataset Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title20 "H2")
  * [ Datatype Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title21 "H2")
  * [ File Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title22 "H2")
  * [ Group Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title23 "H2")
  * [ Link Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title24 "H2")
  * [ Object Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title25 "H2")
  * [ Introspection Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title26 "H2")
  * [ Request (Async) Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title27 "H2")
  * [ Blob Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title28 "H2")
  * [ Token Callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title29 "H2")
  * [ Optional Generic Callback](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title30 "H2")
  * [ New VOL API Routines](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title31 "H1")
  * [ H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title32 "H2")
  * [ H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title33 "H2")
  * [ H5VLconnector_passthru.h](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title34 "H2")
  * [ Appendix A Mapping of VOL Callbacks to HDF5 API Calls](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title35 "H1")
  * [ Appendix B Callback Wrapper API Calls for Passthrough Connector Authors](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title36 "H1")
  * [ Appendix C Native VOL Connector Optional Values By Subclass](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html#title37 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Virtual Object Layer (VOL) Connector Author Guide
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction
The Virtual Object Layer (VOL) is an abstraction layer in the HDF5 library which intercepts all API calls that could potentially access objects in an HDF5 container and forwards those calls to object drivers referred to as _VOL connectors_. The architecture of this feature is described in the [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html) and VOL Architecture and Internals Documentation and will not be duplicated here.
This guide is for people who are interested in developing their own VOL connector for the HDF5 library. It is assumed that the reader has good knowledge of the VOL architecture obtained by reading the VOL architectural design document.
#  Creating a New Connector
##  Overview
Creating a new VOL connector can be a complicated process. You will need to map your storage system to the HDF5 data model through the lens of the VOL and this may involve some impedance mismatch that you will have to work around. The good news is that the HDF5 library has been re-engineered to handle arbitrary, connector-specific data structures via the VOL callbacks, so no knowledge of the library internals is necessary to write a VOL connector.
Writing a VOL connector requires these things: 
  * Decide on library vs plugin vs internal. 
  * Set up your build/test files (CMake, Autotools, etc.). 
  * Fill in some boilerplate information in yourH5VLclasststruct. 
  * Decide how you will perform any necessary initialization needed by your storage system. 
  * Map Storage to HDF5 File Objects 
  * Create implementations for the callbacks you need to support. 
  * Test the connector.


Each of the steps listed above is described in more detail in this section of the document.
The "model then implement" steps can be performed iteratively. You might begin by only supporting files, datasets, and groups and only allowing basic operations on them. In some cases, this may be all that is needed. As your needs grow, you can repeat those steps and increase the connector's HDF5 API coverage at a pace that makes sense for your users.
Also, note that this document only covers writing VOL connectors using the C programming language. It is often possible to write connectors in other programming languages (e.g.; Python) via the language's C interop facilities, but that topic is out of scope for this document.
##  The HDF5 1.12.x VOL Interface Is DEPRECATED
Important changes were made to the VOL interface for HDF5 1.13.0 and, due to binary compatibility issues, these cannot be merged to HDF5 1.12.x. For this reason, VOL connector development should be shifted to target 1.13.0 as no further development of the VOL interface will take place on the 1.12.x branch. Unlike the other development branches of the library, there is no hdf5_1_13 branch - all HDF5 1.13.0 development is taking place in the develop branch of the HDF5 repository and 1.13.x branches will split off from that.
Note also that HDF5 1.13.0 is considered an unstable branch, and the API and file format are subject to change ("unstable" means "not yet finalized", not "buggy"). The VOL feature is under active development and, although it is nearing its final form, may change further before the stable HDF5 1.14.0 release targeted for 2022.
##  VOL-Related HDF5 Header Files
Use of the VOL, including topics such as registration and loading VOL plugins, is described in the [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html).
Public header Files you will need to be familiar with include: 
[H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html) | Public VOL header.   
---|---  
[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html) | Main header for connector authors. Contains definitions for the main VOL struct and callbacks, enum values, etc.   
[H5VLconnector_passthru.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html) | Helper routines for passthrough connector authors.   
[H5VLnative.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lnative_8h.html) | Native VOL connector header. May be useful if your connector will attempt to implement native HDF5 API calls that are handled via the optional callbacks.   
[H5PLextern.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html) | Needed if your connector will be built as a plugin.   
Many VOL connectors are listed on The HDF Group's VOL plugin registration page, located at: [Registered VOL Connectors](https://support.hdfgroup.org/releases/hdf5/documentation/registered_vol_connectors.md). Not all of these VOL connectors are supported by The HDF Group and the level of completeness varies, but the connectors found there can serve as examples of working implementations
##  Library vs Plugin vs Internal
When building a VOL connector, you have several options:
#### Library
The connector can be built as a normal shared or static library. Software that uses your connector will have to link to it just like any other library. This can be convenient since you don't have to deal with plugin paths and searching for the connector at runtime, but it also means that software which uses your connector will have to be built and linked against it.
#### Plugin
You can also build your connector as a dynamically loaded plugin. The mechanism for this is the same mechanism used to dynamically load HDF5 filter plugins. This can allow use of your connector via the VOL environment variable, without modifying the application, but requires your plugin to be discoverable at runtime. See the [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html) for more information about using HDF5 plugins.
To build your connector as a plugin, you will have to include **[H5PLextern.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html)** (a public header distributed with the library) and implement the [H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5) [H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd) calls, both of which are trivial to code up. It also often requires your connector to be built with certain compile/link options. The VOL connector template does all of these things.
The HDF5 library's plugin loading code will call [H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5) to determine the type of plugin(e.g.; filter, VOL) and [H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd) to get the class struct, which allows the library to query the plugin for its name and value to see if it has found a requested plugin. When a match is found, the library will use the class struct to register the connector and map its callbacks.
For the HDF5 library to be able to load an external plugin dynamically, the plugin developer has to define two public routines with the following name and signature: 
[H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) [H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5)(void);
const void *[H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd)(void);
[H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5)
H5PL_type_t H5PLget_plugin_type(void)
[H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd)
const void * H5PLget_plugin_info(void)
[H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f)
H5PL_type_t
**Definition** H5PLpublic.h:33
To show how easy this is to accomplish, here is the complete implementation of those functions in the template VOL connector: 
[H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) [H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5)(void) { return [H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0); }
const void *[H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd)(void) { return &template_class_g; }
[H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0)
@ H5PL_TYPE_VOL
**Definition** H5PLpublic.h:36
[H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5) should return the library type which should always be [H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0). [H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd) should return a pointer to the plugin structure defining the VOL plugin with all the callbacks. For example, consider an external plugin defined as: 
static const [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) H5VL_foo_g = {
2, // version
12345, // value
"foo", // name
...
}
[H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html)
**Definition** H5VLconnector.h:1020
The plugin would implement the two routines as: 
[H5PL_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) [H5PLget_plugin_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5)(void)
{return [H5PL_TYPE_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86fab01b3951d7e978b43e505590a6431bf0);}
const void *[H5PLget_plugin_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd)(void)
{return &H5VL_foo_g;}
#### Internal
Your VOL connector can also be constructed as a part of the HDF5 library. This works in the same way as the stdio and multi virtual file drivers (VFDs) and does not require knowledge of HDF5 internals or use of non-public API calls. You simply have to add your connector's files to the Makefile.am and/or CMakeLists.txt files in the source distribution's src directory. This requires maintaining a private build of the library, though, and is not recommended.
##  Build Files / VOL Template
We have created a template terminal VOL connector that includes both Autotools and CMake build files. The constructed VOL connector includes no real functionality, but can be registered and loaded as a plugin.
The VOL template can be found here: [VOL template](https://github.com/HDFGroup/vol-template)
The purpose of this template is to quickly get you to the point where you can begin filling in the callback functions and writing tests. You can copy this code to your own repository to serve as the basis for your new connector.
A template passthrough VOL is also available. This will be discussed in the section on passthrough connectors.
##  H5VL_class_t Boilerplate
Several fields in the H5VLclasststruct will need to be filled in.
In HDF5 1.13.0, the _version_ field will be 2, indicating the connector targets version 2 of the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct. Version 1 of the struct was never formally released and only available in the develop branch of the HDF5 git repository. Version 0 is used in the deprecated HDF5 1.12.x branch.
Every connector needs a _name_ and _value_. The library will use these when loading and registering the connector (as described in the [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html)), so they should be unique in your ecosystem.
VOL connector values are integers, with a maximum value of 65535. Values from 0 to 255 are reserved for internal use by The HDF Group. The native VOL connector has a value of 0. Values of 256 to 511 are for connector testing and should not be found in the wild. Values of 512 to 65535 are for external connectors.
As is the case with HDF5 filters, The HDF Group can assign you an official VOL connector value. Please contact [help@hdfgroup.org](https://support.hdfgroup.org/documentation/hdf5/latest/_v_o_l__connector.html) for help with this. We currently do not register connector names, though the name you've chosen will appear on the registered VOL connectors page.
As noted above, registered VOL connectors will be listed at: [Registered VOL Connectors](https://support.hdfgroup.org/releases/hdf5/documentation/registered_vol_connectors.md)
A new **conn_version** field has been added to the class struct for 1.13. This field is currently not used by the library so its use is determined by the connector author. Best practices for this field will be determined in the near future and this part of the guide will be updated.
The **cap_flags** field is used to determine the capabilities of the VOL connector. At this time, the use of this field is limited to indicating thread-safety, asynchronous capabilities, and ability to produce native HDF5 files. Supported flags can be found in [H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html). 
// Capability flags for connector
#define H5VL_CAP_FLAG_NONE 0 // No special connector capabilities
#define H5VL_CAP_FLAG_THREADSAFE 0x01 // Connector is threadsafe
#define H5VL_CAP_FLAG_ASYNC 0x02 // Connector performs operations asynchronously
#define H5VL_CAP_FLAG_NATIVE_FILES 0x04 // Connector produces native file format
##  Initialization and Shutdown
You'll need to decide how to perform any initialization and shutdown tasks that are required by your connector. There are initialize and terminate callbacks in the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct to handle this. They are invoked when the connector is registered and unregistered, respectively. The initialize callback can take a VOL initialization property list, so any properties you need for initialization can be applied to it. The HDF5 library currently makes no use of the vipl so there are no default vipl properties.
If this is unsuitable, you may have to create custom connector-specific API calls to handle initialization and termination. It may also be useful to perform operations in a custom API call used to set the VOL connector in the fapl.
The initialization and terminate callbacks: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*initialize)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id); // Connector initialization callback
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*terminate)(void); // Connector termination callback
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
##  Map Storage to HDF5 File Objects
The most difficult part of designing a new VOL connector is determining how to support HDF5 file objects and operations using your storage system. There isn't much specific advice to give here, as each connector will have unique needs, but a forthcoming "tutorial" connector will set up a simple connector and demonstrate this process.
##  Fill In VOL Callbacks
For each file object you support in your connector (including the file itself), you will need to create a data struct to hold whatever file object metadata that are needed by your connector. For example, a data structure for a VOL connector based on text files might have a file struct that contains a file pointer for the text file, buffers used for caching data, etc. Pointers to these data structures are where your connector's state is stored and are returned to the HDF5 library from the create/open/etc. callbacks such as _dataset create_.
Once you have your data structures, you'll need to create your own implementations of the callback functions and map them via your [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct.
##  Handling Optional Operations
Handling optional operations has changed significantly in HDF5 1.13.0. In the past, optional operations were specified using an integer _opt_type_ parameter. This proved to be a problem with pass-through connectors, though, as it was possible to have _opt_type_ clash if two connectors used the same _opt_type_ values.
The new scheme allows a connector to register an optional operation with the library and receive a dynamically-allocated _opt_type_ value for the operation.
The following API calls can be used to manage the optional operations: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9)([H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, const char *op_name, int *op_val);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfind_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab3c45344dc248471076e58dc3f66a9ec)([H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, const char *op_name, int *op_val);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLunregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#aef785a9a3f73d7ce6954ca742e4f8135)([H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, const char *op_name)
[H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9)
herr_t H5VLregister_opt_operation(H5VL_subclass_t subcls, const char *op_name, int *op_val)
[H5VLfind_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab3c45344dc248471076e58dc3f66a9ec)
herr_t H5VLfind_opt_operation(H5VL_subclass_t subcls, const char *op_name, int *op_val)
[H5VLunregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#aef785a9a3f73d7ce6954ca742e4f8135)
herr_t H5VLunregister_opt_operation(H5VL_subclass_t subcls, const char *op_name)
[H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599)
H5VL_subclass_t
**Definition** H5VLpublic.h:183
The _register_ call is used to register an operation for a subclass (file, etc.) and the _opt_type_ parameter that the library assigned to the operation will be returned via the _opt_val_ parameter. This value can then be passed to one of the subclass-specific API calls (listed below). If you need to find an existing optional call's assigned _opt_type_ value by name, you can use the _find_ call.
One recommended way to handle optional calls is to _register_ all the optional calls at startup, saving the values in connector state, then use these cached values in your optional calls. The assigned values should be unregistered using the _unregister_ call when the connector shuts down.
Subclass-specific optional calls: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a932291eed9ef2abfa4c9bbdff88ef073)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a46bf2c58bb032bcae1794aa4cf6f6b59)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdatatype_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a4ecacacc56f7d4e13fda483f901f1d6d)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, hid_tes_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfile_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9234611f30a8d7bc79d1c315c513c9e6)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLgroup_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a942d8bb5cc709547097bd30689be4e60)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ac06a3858e28b503b0d60b13b262a2058)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLobject_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a407bdc0c5d16239a9f9e1c7ffeee6c46)(const char *app_file, const char *app_func, unsigned app_line,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id,
[H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#af22b04c4bd4cfb8b1f319a3ceac86396)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);
[H5VLobject_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a407bdc0c5d16239a9f9e1c7ffeee6c46)
#define H5VLobject_optional_op(...)
**Definition** H5VLconnector.h:1142
[H5VLdataset_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a46bf2c58bb032bcae1794aa4cf6f6b59)
#define H5VLdataset_optional_op(...)
**Definition** H5VLconnector.h:1137
[H5VLdatatype_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a4ecacacc56f7d4e13fda483f901f1d6d)
#define H5VLdatatype_optional_op(...)
**Definition** H5VLconnector.h:1138
[H5VLfile_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9234611f30a8d7bc79d1c315c513c9e6)
#define H5VLfile_optional_op(...)
**Definition** H5VLconnector.h:1139
[H5VLattr_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a932291eed9ef2abfa4c9bbdff88ef073)
#define H5VLattr_optional_op(...)
**Definition** H5VLconnector.h:1136
[H5VLgroup_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a942d8bb5cc709547097bd30689be4e60)
#define H5VLgroup_optional_op(...)
**Definition** H5VLconnector.h:1140
[H5VLlink_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ac06a3858e28b503b0d60b13b262a2058)
#define H5VLlink_optional_op(...)
**Definition** H5VLconnector.h:1141
[H5VLrequest_optional_op](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#af22b04c4bd4cfb8b1f319a3ceac86396)
herr_t H5VLrequest_optional_op(void *req, hid_t connector_id, H5VL_optional_args_t *args)
[H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html)
**Definition** H5VLconnector.h:94
##  Testing Your Connector
At the time of writing, some of the HDF5 library tests have been abstracted out of the library with their native-file-format-only sections removed and added to a VOL test suite available here: [vol-tests](https://github.com/HDFGroup/vol-tests)
This is an evolving set of tests, so see the documentation in that repository for instructions as to its use. You may want to clone and modify and/or extend these tests for use with your own connector.
In the future, we plan to modify the HDF5 test suite that ships with the library to use a future VOL capabilities flags scheme to selectively run tests that a particular connector supports. As this is a large task, it may be some time before that work is complete.
##  Passthrough Connectors
Coming Soon
Note: Passthrough VOL connectors should avoid doing anything with the file in the open and create callbacks except opening it. If connectors need to do anything else they should use the post open callback (H5VL_NATIVE_FILE_POST_OPEN op_type for the file "optional" callback), making sure to perform operations on the file only after passing the post open call down to the terminal connector. The post open callback is made for both file open and file create calls.
##  Asynchronous Operations
Coming Soon
#  VOL Connector Interface Reference
Each VOL connector should be of type [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html): 
_VOL connector class,[H5VLpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lpublic_8h.html)_ typedef struct [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) { /* Overall connector fields & callbacks */ unsigned [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a27fe8e82cf2ce359010cc1d08af2e105);  [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) [value](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a4531586e75e64de257cf51b4b7c50d70);  const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a8f8f80d37794cde9472343e4487ba3eb);  unsigned [conn_version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a6566afc54591b6c64dbec7bdc0bf0b58);  uint64_t [cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#ac4b405cf0a53f734d273232a82baf247);  [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[initialize](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a0a18753f3a24e91b75fd802b81f044a7))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id);  [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[terminate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a3da863f4634c64aeffe46cca882db43d))(void);  /* VOL framework */ H5VL_info_class_t [info_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#afdbe51576a0f6360693757d6738111eb);  H5VL_wrap_class_t [wrap_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a74d541c08c6c71946cc5a1e105ca5e51);  /* Data Model */ H5VL_attr_class_t [attr_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aea95f1ef7bfe9d0a06f5d7ba4d992b33);  H5VL_dataset_class_t [dataset_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a38be37dc57b556f87df46b0ea65c9ae0);  H5VL_datatype_class_t [datatype_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a49a73ac039550d87d60b485727149cce);  H5VL_file_class_t [file_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a94fcae0c3e55fb9da64b99cf5ec0cc05);  H5VL_group_class_t [group_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a6f9e6afe57118c01efbd5805d6f6615d);  H5VL_link_class_t [link_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a3da742717813fedde17f3ef9a739a3a5);  H5VL_object_class_t [object_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aee15298970b7178f78f92de98b7f8211);  /* Infrastructure / Services */ H5VL_introspect_class_t [introspect_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aff81b23d1373b396d5eacda97bb4a6c6);  H5VL_request_class_t [request_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#adc5afcdccffe9dc3608ac8e23e8fea17);  H5VL_blob_class_t [blob_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#aee571d79753251a1d6ebc1073254278c);  H5VL_token_class_t [token_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#acd39460513ffed8d7d3eab4ebab54bb2);  /* Catch-all */ [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, H5VL_optional_args_t *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  } [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html);  
---  
The _version_ field is the version of the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct. This is identical to how the _version_ field is used in the [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html) struct for filters.
The _value_ field is a unique integer identifier that should be between 512 and 65535 for external, non-library connectors.
The _name_ field is a string that uniquely identifies the VOL connector name.
The _conn_version_ is the connector version. This is currently not used by the library.
The _cap_flags_ holds bitwise capability/feature flags that determine which operations and capabilities are supported by a the VOL connector. These fields were enumerated in the previous section.
The _initialize_ field is a function pointer to a routine that a connector implements to set up or initialize access to the connector. Implementing this function by the connector is not required since some connectors do not require any set up to start accessing the connector. In that case, the value of the function pointer should be set to NULL. Connector specific variables that are required to be passed from users should be passed through the VOL initialize property list. Generic properties can be added to this property class for user-defined connectors that cannot modify the HDF5 library to add internal properties. For more information consult the property list reference manual pages.
The _terminate_ field is a function pointer to a routine that a connector implements to terminate or finalize access to the connector. Implementing this function by the connector is not required since some connectors do not require any termination phase to the connector. In that case, the value of the function pointer should be set to NULL.
The rest of the fields in the [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct are "subclasses" that define all the VOL function callbacks that are mapped to from the HDF5 API layer. Those subclasses are categorized into three categories, VOL Framework, Data Model, and Infrastructure / Services.
VOL Framework classes provide functionality for working with the VOL connectors themselves (e.g., working with connector strings) and with wrapping and unwrapping objects for passthrough connectors.
Data Model classes are those that provide functionality for accessing an HDF5 container and objects in that container as defined by the HDF5 data model.
Infrastructure / Service classes are those that provide services for users that are not related to the data model specifically. Asynchronous operations, for example, are a service that most connectors can implement, so we add a class for it in the VOL structure.
If a service becomes generic enough and common among many connectors, a class for it should be added to the VOL structure. However, many connectors can/will provide services that are not shared by other connectors. A good way to support these services is through an optional callback in the VOL structure which can be a hook from the API to the connector that provides those services, passing any necessary arguments needed without the HDF5 library having to worry about supporting that service. A similar API operation to allow users to use that service will be added. This API call would be similar to an "ioctl" call where any kind of operation can be supported and passed down to the connector that has enough knowledge from the user to interpret the type of the operation. All classes and their defined callbacks will be detailed in the following sub-sections.
To handle that large set of API routines, each class in the Data Model category has three generic callbacks, _get_ , _specific_ , and _optional_ to handle the three set of API operations outline above respectively. To handle the varying parameters that can be passed to the callback, each callback will take a struct parameter that includes an enum _get/specific_ or integer _optional_ field indicating the operation and a union of the possible parameters _get/specific_ or void pointer to the parameters _optional_.
The optional args struct used for all optional operations: 
// Struct for all 'optional' callbacks
typedef struct [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) {
int [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html#a6ffc9648ceae4e1ae350df70f097d5f8); // Operation to perform
void *[args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html#add0eb34e0cef9e763462cf9080f9be0a); // Pointer to operation's argument struct
} [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html);
[H5VL_optional_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html#a6ffc9648ceae4e1ae350df70f097d5f8)
int op_type
**Definition** H5VLconnector.h:95
[H5VL_optional_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html#add0eb34e0cef9e763462cf9080f9be0a)
void * args
**Definition** H5VLconnector.h:96
The _opt_type_ member is the value assigned by the library when the optional operation was registered (or _defined_ in the case of the native VOL connector) and the _args_ member is a pointer to the optional operation's parameters (usually passed in as a struct).
Note that this differs from the HDF5 1.12.x scheme, which used _va_lists_.
The _optional_ callback is a free for all callback where anything from the API layer is passed in directly. This callback is used to support connector specific operations in the API that other connectors should or would not know about. More information about types and the arguments for each type will be detailed in the corresponding class arguments.
##  Mapping the API to the Callbacks
The callback interface defined for the VOL has to be general enough to handle all the HDF5 API operations that would access the file. Furthermore, it has to capture future additions to the HDF5 library with little to no changes to the callback interface. Changing the interface often whenever new features are added would be discouraging to connector developers since that would mean reworking their VOL connector structure. To remedy this issue, every callback will contain two parameters: 
  * A data transfer property list (DXPL) which allows that API to put some properties on for the connectors to retrieve if they have to for particular operations, without having to add arguments to the VOL callback function. 
  * A pointer to a request _(void **req)_ to handle asynchronous operations if the HDF5 library adds support for them in future releases. hat pointer is set by the VOL connector to a request object it creates to manage progress on that asynchronous operation. If the _req_ is _NULL_ , that means that the API operation is blocking and so the connector would not execute the operation asynchronously. If the connector does not support asynchronous operations, it needs not to worry about this field and leaves it unset. 


In order to keep the number of the VOL object classes and callbacks concise and readable, it was decided not to have a one-to-one mapping between API operation and callbacks. The parameter names and types will be detailed when describing each callback in their respective sections.
The HDF5 library provides several routines to access an object in the container. For example, to open an attribute on a group object, the user could use [H5Aopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga59863b205b6d93b2145f0fbca49656f7 "Opens an attribute for an object specified by object identifier and attribute name.") and pass the group identifier directly where the attribute needs to be opened. Alternatively, the user could use [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10 "Opens an attribute for an object by object name and attribute name.") or [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d "Opens the nth attribute attached to an object.") to open the attribute, which provides a more flexible way of locating the attribute, whether by a starting object location and a path or an index type and traversal order. All those types of accesses usually map to one VOL callback with a parameter that indicates the access type. In the example of opening an attribute, the three API open routine will map to the same VOL open callback but with a different location parameter. The same applies to all types of routines that have multiple types of accesses. The location parameter is a structure defined in:
_Structure to hold parameters for object locations,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
//
// Structure to hold parameters for object locations.
// either: BY_SELF, BY_NAME, BY_IDX, BY_TOKEN
typedef struct [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) {
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) [obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a81030798b196a9008fdf5b6021205b20); // The object type of the location object
[H5VL_loc_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2) [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a83d3dcb9c6e66dabfd9e83536d086008); // The location type
union { // parameters of the location
[H5VL_loc_by_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__token__t.html) [loc_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#aa5a52c5b2a59a6b6b554d6c7c19ad563);
[H5VL_loc_by_name_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__name__t.html) [loc_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#afe8fbe50a17945ba636392c3210ce0cb);
[H5VL_loc_by_idx_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__idx__t.html) [loc_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a0aac324bbf1f0c6f6d6f2e60f0817ad3);
}[loc_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a7c3f0fc67dc910b999e3ca313a96f5a8);
} [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html)
//
// Types for different ways that objects are located in an
// HDF5 container.
typedef enum [H5VL_loc_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2) {
// starting location is the target object
[H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e),
// location defined by object and path in H5VL_loc_by_name_t
[H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288),
// location defined by object, path, and index in H5VL_loc_by_idx_t
[H5VL_OBJECT_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a4e82e176436dfcd528f38d2e76863a1e),
// location defined by token (e.g. physical address) in H5VL_loc_by_token_t
[H5VL_OBJECT_BY_TOKEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a3b5d0e8f4650eec59183b619174b46cd),
} [H5VL_loc_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2);
typedef struct H5VL_loc_by_name {
const char *name; // The path relative to the starting location
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id; // The link access property list
}[H5VL_loc_by_name_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__name__t.html);
typedef struct H5VL_loc_by_idx {
const char *name; // The path relative to the starting location
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type; // Type of index
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order; // Index traversal order
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n; // Position in index
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id; // The link access property list
}[H5VL_loc_by_idx_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__idx__t.html);
typedef struct H5VL_loc_by_token {
void *token; // arbitrary token (physical address of location in native VOL)
}[H5VL_loc_by_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__token__t.html);
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b)
H5I_type_t
**Definition** H5Ipublic.h:34
[H5VL_loc_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2)
H5VL_loc_type_t
**Definition** H5VLconnector.h:50
[H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288)
@ H5VL_OBJECT_BY_NAME
**Definition** H5VLconnector.h:52
[H5VL_OBJECT_BY_TOKEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a3b5d0e8f4650eec59183b619174b46cd)
@ H5VL_OBJECT_BY_TOKEN
**Definition** H5VLconnector.h:54
[H5VL_OBJECT_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a4e82e176436dfcd528f38d2e76863a1e)
@ H5VL_OBJECT_BY_IDX
**Definition** H5VLconnector.h:53
[H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e)
@ H5VL_OBJECT_BY_SELF
**Definition** H5VLconnector.h:51
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9)
H5_iter_order_t
**Definition** H5public.h:379
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3)
H5_index_t
**Definition** H5public.h:402
[H5VL_loc_by_idx_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__idx__t.html)
**Definition** H5VLconnector.h:62
[H5VL_loc_by_name_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__name__t.html)
**Definition** H5VLconnector.h:57
[H5VL_loc_by_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__by__token__t.html)
**Definition** H5VLconnector.h:70
[H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html)
**Definition** H5VLconnector.h:83
[H5VL_loc_params_t::loc_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a0aac324bbf1f0c6f6d6f2e60f0817ad3)
H5VL_loc_by_idx_t loc_by_idx
**Definition** H5VLconnector.h:89
[H5VL_loc_params_t::loc_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a7c3f0fc67dc910b999e3ca313a96f5a8)
union H5VL_loc_params_t::@253232363157051151176002046324213315140375116335 loc_data
[H5VL_loc_params_t::obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a81030798b196a9008fdf5b6021205b20)
H5I_type_t obj_type
**Definition** H5VLconnector.h:84
[H5VL_loc_params_t::type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#a83d3dcb9c6e66dabfd9e83536d086008)
H5VL_loc_type_t type
**Definition** H5VLconnector.h:85
[H5VL_loc_params_t::loc_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#aa5a52c5b2a59a6b6b554d6c7c19ad563)
H5VL_loc_by_token_t loc_by_token
**Definition** H5VLconnector.h:87
[H5VL_loc_params_t::loc_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html#afe8fbe50a17945ba636392c3210ce0cb)
H5VL_loc_by_name_t loc_by_name
**Definition** H5VLconnector.h:88
##  Connector Information Callbacks
This section's callbacks involve the connector-specific information that will be associated with the VOL in the fapl via **H5Pset_fapl_ <name>** et al. This data is copied into the fapl so the library needs these functions to manage this in a way that prevents resource leaks.
The _to_str_ and _from_str_ callbacks are used to convert the connector-specific data to and from a configuration string. There is no official way to construct VOL configuration strings, so the format used (JSON, XML, getopt-style processing, etc.) is up to the connector author. These connector configuration strings can be used to set up a VOL connector via mechanisms like command-line parameters and environment variables.
_Info class for connector information routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
// VOL connector info fields & callbacks 
typedef struct [H5VL_info_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html) {
size_t [size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a854352f53b148adc24983a58a1866d66); // Size of the VOL info
void *(*copy)(const void *info); // Callback to create a copy of the VOL info
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[cmp](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#ac9e6950dcb458c72df4857000306e4ea))(int *cmp_value, const void *info1, const void *info2); // Callback to compare VOL info
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#aaef036b59719520efc816a744a158818))(void *info); // Callback to release a VOL info
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[to_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a9b52b286e232368e3c2a5846481d2d93))(const void *info, char **str); // Callback to serialize connector's info into a string
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[from_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a61855a9ef842935572f989e47be760bd))(const char *str, void **info); // Callback to deserialize a string into connector's info
} [H5VL_info_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html);
[H5VL_info_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html)
**Definition** H5VLconnector.h:845
[H5VL_info_class_t::from_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a61855a9ef842935572f989e47be760bd)
herr_t(* from_str)(const char *str, void **info)
**Definition** H5VLconnector.h:851
[H5VL_info_class_t::size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a854352f53b148adc24983a58a1866d66)
size_t size
**Definition** H5VLconnector.h:846
[H5VL_info_class_t::to_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#a9b52b286e232368e3c2a5846481d2d93)
herr_t(* to_str)(const void *info, char **str)
**Definition** H5VLconnector.h:850
[H5VL_info_class_t::free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#aaef036b59719520efc816a744a158818)
herr_t(* free)(void *info)
**Definition** H5VLconnector.h:849
[H5VL_info_class_t::cmp](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__info__class__t.html#ac9e6950dcb458c72df4857000306e4ea)
herr_t(* cmp)(int *cmp_value, const void *info1, const void *info2)
**Definition** H5VLconnector.h:848
###  info: size
The _size_ field indicates the size required to store any special information that the connector needs.
If the connector requires no special information, set this field to zero. 
Signature:   
---  
size_t size;  
###  info: copy
The _copy_ callback is invoked when the connector is selected for use with **H5Pset_fapl_ <name>**, the connector-specific set call, etc. Where possible, the information should be deep copied in such a way that the original data can be freed. 
Signature:   
---  
void * (*copy)(const void *info);  
Arguments:   
info (IN): The connector-specific info to copy.  
###  info: cmp
The _cmp_ callback is used to determine if two connector-specific data structs are identical and helps the library manage connector resources. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*cmp)(int *cmp_value, const void *info1, const void *info2);  
Arguments:   
cmp_value (OUT): A strcmp-like compare value. info1 (IN): The 1st connector-specific info to copy. info2 (IN): The 2nd connector-specific info to copy.  
###  info: free
The _free_ callback is used to clean up the connector-specific information that was copied when set in the fapl via the _copy_ callback. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*free)(void *info);  
Arguments:   
info (IN): The connector-specific info to free.  
###  info: to_str
The _to_str_ callback converts a connector-specific information structure to a connector-specific configuration string. It is the opposite of the _from_str_ callback. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*to_str)(const void *info, char **str);  
Arguments:   
info (IN): The connector-specific info to convert to a configuration string. str (OUT): The constructed configuration string.  
###  info: from_str
The _from_str_ callback converts a connector-specific configuration string to a connector-specific information structure. It is the opposite of the _to_str_ callback. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*from_str)(const char *str, void **info);  
Arguments:   
str (IN): The connector-specific configuration string. info (OUT): The connector-specific info generated from the configuration string.  
##  Object Wrap Callbacks
The object wrap callbacks are used by passthrough connectors to wrap/unwrap objects and contexts when passing them up and down the VOL chain. Each passthrough VOL must define an object wrapping structure and a wrap context. The object wrapping structure should contain the information necessary to recursively unwrap the object - at a minimum, this is the object provided from and the ID of the next connector in the stack. The wrap context should contain the information necessary to recursively wrap the object - at a minimum, this is the ID and the wrap context of the next VOL connector in the stack.
Each callback should use _H5VL <callback>_ to recursively invoke the same callback in all lower passthrough VOl connectors.
_Wrap class for object wrapping routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_wrap_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html) {
void *(*get_object)(const void *obj); // Callback to retrieve underlying object
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html#a9fa016d5f49513d81a76662102129c02))(const void *obj, void **wrap_ctx); // Callback to retrieve the object wrapping context for the connector
void *(*wrap_object)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, void *wrap_ctx); // Callback to wrap a library object
void *(*unwrap_object)(void *obj); // Callback to unwrap a library object
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[free_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html#a64eb31204961fce4a5e2d7a38f610e19))(void *wrap_ctx); // Callback to release the object wrapping context for the connector
} [H5VL_wrap_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html);
[H5VL_wrap_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html)
**Definition** H5VLconnector.h:858
[H5VL_wrap_class_t::free_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html#a64eb31204961fce4a5e2d7a38f610e19)
herr_t(* free_wrap_ctx)(void *wrap_ctx)
**Definition** H5VLconnector.h:866
[H5VL_wrap_class_t::get_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__wrap__class__t.html#a9fa016d5f49513d81a76662102129c02)
herr_t(* get_wrap_ctx)(const void *obj, void **wrap_ctx)
**Definition** H5VLconnector.h:860
###  wrap: get_object
Retrieves the underlying object from a wrapped object. Should return a pointer to the underlying object belonging to the terminal VOL connector.
This will generally be done by unwrapping this VOL's object wrapping structure, before recursively calling _H5VLget_object_ to invoke the _get_object_ callback for all lower VOL connectors which define it. _H5VLget_object_ requires the object returned by the next VOL and the next VOL's ID. Both of these fields should be stored by the VOL somehow, generally on the wrapped object structure.
This callback should not cleanup or modify the provided object wrapping structure. 
Signature:   
---  
void * (*get_object)(const void *obj);  
Arguments:   
obj (IN): The object to be unwrapped.  
###  wrap: get_wrap_ctx
Get a VOL connector's object wrapping context.
The context should be returned in dynamically allocated memory under _*wrap_ctx_. Any resources this callback allocates should be freed within _free_wrap_ctx_. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get_wrap_ctx)(const void *obj, void **wrap_ctx);  
Arguments:   
obj (IN): Object wrapped by this VOL connector, for which we need a context. wrap_ctx (OUT): Context.  
###  wrap: wrap_object
Asks a connector to wrap an underlying object. This callback should use _H5VLwrap_object_ to recursively have the object wrapped by all lower VOL connectors before performing its own wrapping.
The wrapped object should provide the information necessary for _unwrap_object_ to recursively unwrap it - at a minimum, the object provided from and the ID of the next VOL connector. 
Signature:   
---  
void * (*wrap_object)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, void *wrap_ctx);  
Arguments:   
obj (IN): The object to be wrapped. obj_type (IN): Object type (see H5Ipublic.h). wrap_ctx (IN): Context.  
###  wrap: unwrap_object
Unwrap an object from connector. Any resources allocated during _wrap_object_ should be released and cleaned up here.
This callback should clean up this VOL's object wrapping structure before recursively invoking _H5VLunwrap_object_.
Signature:   
---  
void * (*unwrap_object)(void *obj);  
Arguments:   
obj (IN): Object to be unwrapped.  
###  wrap: free_wrap_ctx
Release a VOL connector's object wrapping context. This should free any resources allocated during _get_wrap_ctx_ , and recursively invoke _H5VLfree_wrap_ctx_ to execute the free callback for the lower VOL connectors in the stack. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*free_wrap_ctx)(void *wrap_ctx);  
Arguments:   
wrap_ctx (IN): Context to be freed.  
##  The Attribute Function Callbacks
The attribute API routines ([Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html)) allow HDF5 users to create and manage HDF5 attributes. All the [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) API routines that modify the HDF5 container map to one of the attribute callback routines in this class that the connector needs to implement.
_Structure for attribute callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_attr_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html) {
void *(*create)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
void *(*open)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[read](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a545cd43f5be5217d472e63c850309d1d))(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[write](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a0fcaa82af4caaf3197036eaff5a23cb5))(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#ac18dc606174e4606dcfccfe7b9303272))(void *obj, [H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a646b7141d2d0471d03190ddd82117a5c))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a6994e0695e38d086ff5cb3b95a305bf1))(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_attr_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html);
[H5VL_attr_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html)
**Definition** H5VLconnector.h:871
[H5VL_attr_class_t::write](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a0fcaa82af4caaf3197036eaff5a23cb5)
herr_t(* write)(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:877
[H5VL_attr_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a533cec569e2019874f02dd9552bcb68e)
herr_t(* optional)(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:881
[H5VL_attr_class_t::read](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a545cd43f5be5217d472e63c850309d1d)
herr_t(* read)(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:876
[H5VL_attr_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a646b7141d2d0471d03190ddd82117a5c)
herr_t(* specific)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:879
[H5VL_attr_class_t::close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#a6994e0695e38d086ff5cb3b95a305bf1)
herr_t(* close)(void *attr, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:882
[H5VL_attr_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__class__t.html#ac18dc606174e4606dcfccfe7b9303272)
herr_t(* get)(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:878
[H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html)
**Definition** H5VLconnector.h:125
[H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html)
**Definition** H5VLconnector.h:184
###  attr: create
The _create_ callback in the attribute class creates an attribute object in the container of the location object and returns a pointer to the attribute structure containing information to access the attribute in future calls. 
Signature:   
---  
void *(*create)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the attribute needs to be created or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". attr_name (IN): The name of the attribute to be created. type_id (IN): The datatype of the attribute. space_id (IN): The dataspace of the attribute. acpl_id (IN): The attribute creation property list. aapl_id (IN): The attribute access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  attr: open
The _open_ callback in the attribute class opens an attribute object in the container of the location object and returns a pointer to the attribute structure containing information to access the attribute in future calls.
Signature:   
---  
void *(*open)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the attribute needs to be opened or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". attr_name (IN): The name of the attribute to be opened. aapl_id (IN): The attribute access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  attr: read
The _read_ callback in the attribute class reads data from the attribute object and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*read)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
attr (IN): Pointer to the attribute object. mem_type_id (IN): The memory datatype of the attribute. buf (OUT): Data buffer to be read into. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  attr: write
The _write_ callback in the attribute class writes data to the attribute object and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*write)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
attr (IN): Pointer to the attribute object. mem_type_id (IN): The memory datatype of the attribute. buf (IN): Data buffer to be written. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  attr: get
The _get_ callback in the attribute class retrieves information about the attribute as specified in the _get_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, [H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): An attribute or location object where information needs to be retrieved from. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for attribute 'get' operations */
typedef enum [H5VL_attr_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386) {
[H5VL_ATTR_GET_ACPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a28c828cecf474623e8e0103a1c03a119), /* creation property list */
[H5VL_ATTR_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a245c8729545d02ae0b9dcdf3598c17e2), /* info */
[H5VL_ATTR_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a3566d35892e514ccab13b48725d6f6ef), /* access property list */
[H5VL_ATTR_GET_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a257bbe582665afa336fe112b78d5fdbc), /* dataspace */
[H5VL_ATTR_GET_STORAGE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386ab3ad72c306c4a1722505e4b1ad4a415a), /* storage size */
[H5VL_ATTR_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a02ea42f42ca98b712b63efb52aacbfd1) /* datatype */
} [H5VL_attr_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386);
/* Parameters for attribute 'get_name' operation */
typedef struct [H5VL_attr_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html) {
[H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) [loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad); /* Location parameters for object access */
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of attribute name buffer */
char *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1); /* Buffer for attribute name (OUT) */
size_t *[attr_name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#af5465ff35213cc3d895523392f07f8fe); /* Actual length of attribute name (OUT) */
} [H5VL_attr_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html);
/* Parameters for attribute 'get_info' operation */
typedef struct [H5VL_attr_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html) {
H5VL_loc_params_t [loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad); /* Location parameters for object access */
const char *[attr_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a7f2b640f7bd6d1c0a810c2824ad056c7); /* Attribute name (for get_info_by_name) */
H5A_info_t *[ainfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a2075ca161c7fd1ed39ca4ec2dbec1c20); /* Attribute info (OUT) */
} [H5VL_attr_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html);
/* Parameters for attribute 'get' operations */
typedef struct [H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html) {
[H5VL_attr_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a267792a7407e0ff0dfc9b5c1da575117); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_ATTR_GET_ACPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [acpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#ac7e3d57f1868bfe6b8f56f985d94b487); /* Attribute creation property list ID (OUT) */
} [get_acpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a5573d480417f45445282b82dea3ef3bc);
/* H5VL_ATTR_GET_INFO */
H5VL_attr_get_info_args_t [get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a2fcdf813baa9148a32e271131bb4cc3a); /* Attribute info */
/* H5VL_ATTR_GET_NAME */
H5VL_attr_get_name_args_t [get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#afa0c39c2b18a3edc59ba82462fc1450d); /* Attribute name */
/* H5VL_ATTR_GET_SPACE */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [space_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a0327c24b4b50fe1c105339cb7ce599ab); /* Dataspace ID (OUT) */
} [get_space](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a736cc0e037ed24122a6fab176e9756e9);
/* H5VL_ATTR_GET_STORAGE_SIZE */
struct {
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *[data_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a5817cccb2c952f74883ddea00bc999b4); /* Size of attribute in file (OUT) */
} [get_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a7469ad909c5fa8de20e3ca8b624f9d30);
/* H5VL_ATTR_GET_TYPE */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac); /* Datatype ID (OUT) */
} [get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a975934a63108c733242aeff3dc857d2e);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a2b679afa1d6be0f60e69fc47078ebc52);
} [H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html);
[H5VL_attr_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386)
H5VL_attr_get_t
**Definition** H5VLconnector.h:100
[H5VL_ATTR_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a02ea42f42ca98b712b63efb52aacbfd1)
@ H5VL_ATTR_GET_TYPE
**Definition** H5VLconnector.h:106
[H5VL_ATTR_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a245c8729545d02ae0b9dcdf3598c17e2)
@ H5VL_ATTR_GET_INFO
**Definition** H5VLconnector.h:102
[H5VL_ATTR_GET_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a257bbe582665afa336fe112b78d5fdbc)
@ H5VL_ATTR_GET_SPACE
**Definition** H5VLconnector.h:104
[H5VL_ATTR_GET_ACPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a28c828cecf474623e8e0103a1c03a119)
@ H5VL_ATTR_GET_ACPL
**Definition** H5VLconnector.h:101
[H5VL_ATTR_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386a3566d35892e514ccab13b48725d6f6ef)
@ H5VL_ATTR_GET_NAME
**Definition** H5VLconnector.h:103
[H5VL_ATTR_GET_STORAGE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab1ca6e780261f54b22ab29df7db17386ab3ad72c306c4a1722505e4b1ad4a415a)
@ H5VL_ATTR_GET_STORAGE_SIZE
**Definition** H5VLconnector.h:105
[H5VL_attr_get_args_t::space_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a0327c24b4b50fe1c105339cb7ce599ab)
hid_t space_id
**Definition** H5VLconnector.h:143
[H5VL_attr_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a267792a7407e0ff0dfc9b5c1da575117)
H5VL_attr_get_t op_type
**Definition** H5VLconnector.h:126
[H5VL_attr_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a2b679afa1d6be0f60e69fc47078ebc52)
union H5VL_attr_get_args_t::@100374311024317025337351276333353044314342114073 args
[H5VL_attr_get_args_t::get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a2fcdf813baa9148a32e271131bb4cc3a)
H5VL_attr_get_info_args_t get_info
**Definition** H5VLconnector.h:136
[H5VL_attr_get_args_t::get_acpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a5573d480417f45445282b82dea3ef3bc)
struct H5VL_attr_get_args_t::@100374311024317025337351276333353044314342114073::@067117016065005351362166264200113355036273274360 get_acpl
[H5VL_attr_get_args_t::data_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a5817cccb2c952f74883ddea00bc999b4)
hsize_t * data_size
**Definition** H5VLconnector.h:148
[H5VL_attr_get_args_t::get_space](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a736cc0e037ed24122a6fab176e9756e9)
struct H5VL_attr_get_args_t::@100374311024317025337351276333353044314342114073::@357061065323037200370270036054224265260002204330 get_space
[H5VL_attr_get_args_t::get_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a7469ad909c5fa8de20e3ca8b624f9d30)
struct H5VL_attr_get_args_t::@100374311024317025337351276333353044314342114073::@205161204202333344262357165125212102227046163263 get_storage_size
[H5VL_attr_get_args_t::type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac)
hid_t type_id
**Definition** H5VLconnector.h:153
[H5VL_attr_get_args_t::get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#a975934a63108c733242aeff3dc857d2e)
struct H5VL_attr_get_args_t::@100374311024317025337351276333353044314342114073::@320252006150362271205374033246275343017306055340 get_type
[H5VL_attr_get_args_t::acpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#ac7e3d57f1868bfe6b8f56f985d94b487)
hid_t acpl_id
**Definition** H5VLconnector.h:132
[H5VL_attr_get_args_t::get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html#afa0c39c2b18a3edc59ba82462fc1450d)
H5VL_attr_get_name_args_t get_name
**Definition** H5VLconnector.h:139
[H5VL_attr_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html)
**Definition** H5VLconnector.h:118
[H5VL_attr_get_info_args_t::ainfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a2075ca161c7fd1ed39ca4ec2dbec1c20)
H5A_info_t * ainfo
**Definition** H5VLconnector.h:121
[H5VL_attr_get_info_args_t::loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad)
H5VL_loc_params_t loc_params
**Definition** H5VLconnector.h:119
[H5VL_attr_get_info_args_t::attr_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__info__args__t.html#a7f2b640f7bd6d1c0a810c2824ad056c7)
const char * attr_name
**Definition** H5VLconnector.h:120
[H5VL_attr_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html)
**Definition** H5VLconnector.h:110
[H5VL_attr_get_name_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1)
char * buf
**Definition** H5VLconnector.h:113
[H5VL_attr_get_name_args_t::loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad)
H5VL_loc_params_t loc_params
**Definition** H5VLconnector.h:111
[H5VL_attr_get_name_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:112
[H5VL_attr_get_name_args_t::attr_name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__name__args__t.html#af5465ff35213cc3d895523392f07f8fe)
size_t * attr_name_len
**Definition** H5VLconnector.h:114
###  attr: specific
The _specific_ callback in the attribute class implements specific operations on HDF5 attributes as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The location object where the operation needs to happen. loc_params (IN): A pointer to the location parameters as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for attribute 'specific' operation */
typedef enum [H5VL_attr_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1) {
[H5VL_ATTR_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1ae01237a4f14a54465e5a3a04f18a5c1d), /* H5Adelete(_by_name) */
[H5VL_ATTR_DELETE_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1acd8869b073aceaa7a843d5d8bbf23e6b), /* H5Adelete_by_idx */
[H5VL_ATTR_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1a4c65cce04f97b2affbd8cb27b3161a04), /* H5Aexists(_by_name) */
[H5VL_ATTR_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1ab28ca25824a3879d755db3845f70ccad), /* H5Aiterate(_by_name) */
[H5VL_ATTR_RENAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1a6d0e93c4d1a0c4bd45211d7e0b262269) /* H5Arename(_by_name) */
} [H5VL_attr_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1);
/* Parameters for attribute 'iterate' operation */
typedef struct [H5VL_attr_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html) {
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) [idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0); /* Type of index to iterate over */
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) [order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb); /* Order of index iteration */
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *[idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#a88b05afb3975738c182e83d77fd6c21d); /* Start/stop iteration index (IN/OUT) */
[H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74) [op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#afc42db8786ce730626f2ef7190e86fb6); /* Iteration callback function */
void *[op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#aa1649fef45b8c728cf110597e5d51f45); /* Iteration callback context */
} [H5VL_attr_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html);
/* Parameters for attribute 'delete_by_idx' operation */
typedef struct [H5VL_attr_delete_by_idx_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html) {
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) [idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0); /* Type of index to iterate over */
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) [order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb); /* Order of index iteration */
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [n](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#a615b5c028cce1e117eae499a06a51663); /* Iteration index */
} [H5VL_attr_delete_by_idx_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html);
/* Parameters for attribute 'specific' operations */
typedef struct [H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html) {
[H5VL_attr_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#ad5e60d51e3bf80f56ba3b895236f8d08); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_ATTR_DELETE */
struct {
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a8f8f80d37794cde9472343e4487ba3eb); /* Name of attribute to delete */
} [del](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a4b4425977bb657b482d152a9569b8d6a);
/* H5VL_ATTR_DELETE_BY_IDX */
H5VL_attr_delete_by_idx_args_t [delete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a7d81e7a1bc7fc97abf5547985104f136);
/* H5VL_ATTR_EXISTS */
struct {
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a8f8f80d37794cde9472343e4487ba3eb); /* Name of attribute to check */
bool *[exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c); /* Whether attribute exists (OUT) */
} [exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c);
/* H5VL_ATTR_ITER */
H5VL_attr_iterate_args_t [iterate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a9f8d2ff2f716d444e4e42eed10ec2c04);
/* H5VL_ATTR_RENAME */
struct {
const char *[old_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a242c1437f4a5d06f640fa56f1441b808); /* Name of attribute to rename */
const char *[new_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#aa7bc4190119c5ebb025fe673fe7b198b); /* New attribute name */
} [rename](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a2fd231239600126c6bee6dc82245cb18);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#ae0164729bfa93383d26746d1251b21e7);
} [H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html);
[H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)
herr_t(* H5A_operator2_t)(hid_t location_id, const char *attr_name, const H5A_info_t *ainfo, void *op_data)
**Definition** H5Apublic.h:60
[H5VL_attr_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1)
H5VL_attr_specific_t
**Definition** H5VLconnector.h:159
[H5VL_ATTR_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1a4c65cce04f97b2affbd8cb27b3161a04)
@ H5VL_ATTR_EXISTS
**Definition** H5VLconnector.h:162
[H5VL_ATTR_RENAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1a6d0e93c4d1a0c4bd45211d7e0b262269)
@ H5VL_ATTR_RENAME
**Definition** H5VLconnector.h:164
[H5VL_ATTR_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1ab28ca25824a3879d755db3845f70ccad)
@ H5VL_ATTR_ITER
**Definition** H5VLconnector.h:163
[H5VL_ATTR_DELETE_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1acd8869b073aceaa7a843d5d8bbf23e6b)
@ H5VL_ATTR_DELETE_BY_IDX
**Definition** H5VLconnector.h:161
[H5VL_ATTR_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab8f6ab518f80d1afedd8b16ac3454ac1ae01237a4f14a54465e5a3a04f18a5c1d)
@ H5VL_ATTR_DELETE
**Definition** H5VLconnector.h:160
[H5VL_attr_delete_by_idx_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html)
**Definition** H5VLconnector.h:177
[H5VL_attr_delete_by_idx_args_t::idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0)
H5_index_t idx_type
**Definition** H5VLconnector.h:178
[H5VL_attr_delete_by_idx_args_t::n](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#a615b5c028cce1e117eae499a06a51663)
hsize_t n
**Definition** H5VLconnector.h:180
[H5VL_attr_delete_by_idx_args_t::order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__delete__by__idx__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb)
H5_iter_order_t order
**Definition** H5VLconnector.h:179
[H5VL_attr_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html)
**Definition** H5VLconnector.h:168
[H5VL_attr_iterate_args_t::idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0)
H5_index_t idx_type
**Definition** H5VLconnector.h:169
[H5VL_attr_iterate_args_t::idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#a88b05afb3975738c182e83d77fd6c21d)
hsize_t * idx
**Definition** H5VLconnector.h:171
[H5VL_attr_iterate_args_t::op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#aa1649fef45b8c728cf110597e5d51f45)
void * op_data
**Definition** H5VLconnector.h:173
[H5VL_attr_iterate_args_t::order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb)
H5_iter_order_t order
**Definition** H5VLconnector.h:170
[H5VL_attr_iterate_args_t::op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__iterate__args__t.html#afc42db8786ce730626f2ef7190e86fb6)
H5A_operator2_t op
**Definition** H5VLconnector.h:172
[H5VL_attr_specific_args_t::old_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a242c1437f4a5d06f640fa56f1441b808)
const char * old_name
**Definition** H5VLconnector.h:208
[H5VL_attr_specific_args_t::rename](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a2fd231239600126c6bee6dc82245cb18)
struct H5VL_attr_specific_args_t::@070074341170002175041205164317267231205371227331::@242112206330127346145113136041276270171361145231 rename
[H5VL_attr_specific_args_t::del](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a4b4425977bb657b482d152a9569b8d6a)
struct H5VL_attr_specific_args_t::@070074341170002175041205164317267231205371227331::@300131260047166104256273140142342331177171136246 del
[H5VL_attr_specific_args_t::delete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a7d81e7a1bc7fc97abf5547985104f136)
H5VL_attr_delete_by_idx_args_t delete_by_idx
**Definition** H5VLconnector.h:195
[H5VL_attr_specific_args_t::name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a8f8f80d37794cde9472343e4487ba3eb)
const char * name
**Definition** H5VLconnector.h:191
[H5VL_attr_specific_args_t::exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c)
bool * exists
**Definition** H5VLconnector.h:200
[H5VL_attr_specific_args_t::iterate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#a9f8d2ff2f716d444e4e42eed10ec2c04)
H5VL_attr_iterate_args_t iterate
**Definition** H5VLconnector.h:204
[H5VL_attr_specific_args_t::new_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#aa7bc4190119c5ebb025fe673fe7b198b)
const char * new_name
**Definition** H5VLconnector.h:209
[H5VL_attr_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#ad5e60d51e3bf80f56ba3b895236f8d08)
H5VL_attr_specific_t op_type
**Definition** H5VLconnector.h:185
[H5VL_attr_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html#ae0164729bfa93383d26746d1251b21e7)
union H5VL_attr_specific_args_t::@070074341170002175041205164317267231205371227331 args
###  attr: optional
The _optional_ callback in the attribute class implements connector specific operations on an HDF5 attribute. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  attr: close
The _close_ callback in the attribute class terminates access to the attribute object and free all resources it was consuming, and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
attr (IN): Pointer to the attribute object. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
##  Dataset Callbacks
The dataset API routines ([Datasets (H5D)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html)) allow HDF5 users to create and manage HDF5 datasets. All the [Datasets (H5D)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html) API routines that modify the HDF5 container map to one of the dataset callback routines in this class that the connector needs to implement.
_Structure for dataset callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_dataset_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html) {
void *(*create)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
void *(*open)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[read](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a7eb6727202a2bfc21d35d1de2d9752e2))(size_t count, void *dset[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[],
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf[], void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[write](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a85ef69e00e7fa8389d5e021a5a4b5d09))(size_t count, void *dset[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[],
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf[], void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a90742926de704a6829c191ab19b4b422))(void *obj, [H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#af88a7553cd7b00f16a2a293b4f441a21))(void *obj, [H5VL_dataset_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a6d5c8d0bd557911bbb792caa52c1a4c1))(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_dataset_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html);
[H5VL_dataset_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html)
**Definition** H5VLconnector.h:886
[H5VL_dataset_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a533cec569e2019874f02dd9552bcb68e)
herr_t(* optional)(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:897
[H5VL_dataset_class_t::close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a6d5c8d0bd557911bbb792caa52c1a4c1)
herr_t(* close)(void *dset, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:898
[H5VL_dataset_class_t::read](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a7eb6727202a2bfc21d35d1de2d9752e2)
herr_t(* read)(size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[], hid_t dxpl_id, void *buf[], void **req)
**Definition** H5VLconnector.h:891
[H5VL_dataset_class_t::write](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a85ef69e00e7fa8389d5e021a5a4b5d09)
herr_t(* write)(size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[], hid_t dxpl_id, const void *buf[], void **req)
**Definition** H5VLconnector.h:893
[H5VL_dataset_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#a90742926de704a6829c191ab19b4b422)
herr_t(* get)(void *obj, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:895
[H5VL_dataset_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__class__t.html#af88a7553cd7b00f16a2a293b4f441a21)
herr_t(* specific)(void *obj, H5VL_dataset_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:896
[H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html)
**Definition** H5VLconnector.h:228
[H5VL_dataset_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html)
**Definition** H5VLconnector.h:273
###  dataset: create
The _create_ callback in the dataset class creates a dataset object in the container of the location object and returns a pointer to the dataset structure containing information to access the dataset in future calls. 
Signature:   
---  
void *(*create)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the dataset needs to be created or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. name (IN): The name of the dataset to be created. lcpl_id (IN): The link creation property list. type_id (IN): The datatype of the dataset. space_id (IN): The dataspace of the dataset. dcpl_id (IN): The dataset creation property list. dapl_id (IN): The dataset access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  dataset: open
The _open_ callback in the dataset class opens a dataset object in the container of the location object and returns a pointer to the dataset structure containing information to access the dataset in future calls. 
Signature:   
---  
void *(*open)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the dataset needs to be opened or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. name (IN): The name of the dataset to be opened. dapl_id (IN): The dataset access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  dataset: read
The _read_ callback in the dataset class reads data from the dataset object and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*read)(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf, void **req);  
Arguments:   
dset (IN): Pointer to the dataset object. mem_type_id (IN): The memory datatype of the data. mem_space_id (IN): The memory dataspace selection. file_space_id (IN): The file dataspace selection. dxpl_id (IN): The data transfer property list. buf (OUT): Data buffer to be read into. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  dataset: write
The _write_ callback in the dataset class writes data to the dataset object and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*write)(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf, void **req);  
Arguments:   
dset (IN): Pointer to the dataset object. mem_type_id (IN): The memory datatype of the data. mem_space_id (IN): The memory dataspace selection. file_space_id (IN): The file dataspace selection. dxpl_id (IN): The data transfer property list. buf (IN): Data buffer to be written from. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  dataset: get
The _get_ callback in the dataset class retrieves information about the dataset as specified in the _get_type_ parameter.It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *dset, [H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
dset (IN): The dataset object where information needs to be retrieved from. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for dataset 'get' operation */
typedef enum [H5VL_dataset_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372) {
[H5VL_DATASET_GET_DAPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a04105bc0c224fc85120a6794b1141211), /* access property list */
[H5VL_DATASET_GET_DCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a5c35e81f46b4a93c0efc89e2a0b97550), /* creation property list */
[H5VL_DATASET_GET_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a31d5a7839c3c4ece749c4d4b5e5f2ea9), /* dataspace */
[H5VL_DATASET_GET_SPACE_STATUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a7514a11451c7d583ac8a247e36e62906), /* space status */
[H5VL_DATASET_GET_STORAGE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a64752afc213ac4e5fe4d954b78442adc), /* storage size */
[H5VL_DATASET_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372aa83d7cc17db73e5eb310687f3e2d3a9f) /* datatype */
} [H5VL_dataset_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372);
/* Parameters for dataset 'get' operations */
typedef struct [H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html) {
[H5VL_dataset_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#acc2d43af01dd0431426e99cb3b7058d2); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_DATASET_GET_DAPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [dapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a6ee841fff797bec3dab658a3e113b747); /* Dataset access property list ID (OUT) */
} [get_dapl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#af8c5b98e5690e7e7db06f29f6593b8e9);
/* H5VL_DATASET_GET_DCPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [dcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a3b10a61cf12d9536d1d91ae50313a5da); /* Dataset creation property list ID (OUT) */
} [get_dcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#af400cdfe9e1a85b8ac50828ad9c5c8f4);
/* H5VL_DATASET_GET_SPACE */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [space_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a0327c24b4b50fe1c105339cb7ce599ab); /* Dataspace ID (OUT) */
} [get_space](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#aa0c6e1eb6a730ce87ef4aeb5c2946f5b);
/* H5VL_DATASET_GET_SPACE_STATUS */
struct {
[H5D_space_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a28e60d50e4eaeef27130829f66e39c7a) *[status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#add2a21e9cdd0ea1b51f4956858b999d5); /* Storage space allocation status (OUT) */
} [get_space_status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ad09bb951f5850c3e8381f2c34ac43214);
/* H5VL_DATASET_GET_STORAGE_SIZE */
struct {
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *[storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#aa1b9b69e3b88a073d07acbd7a2a96f31); /* Size of dataset's storage (OUT) */
} [get_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ad9772107bfc2eaced2ccbfd8f8f7f968);
/* H5VL_DATASET_GET_TYPE */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac); /* Datatype ID (OUT) */
} [get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ac28d98e8c75a9d519ab6d0526ca8a548);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ab4ffab5d9944ce8debc82cfd8d15953c);
} [H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html);
[H5D_space_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a28e60d50e4eaeef27130829f66e39c7a)
H5D_space_status_t
**Definition** H5Dpublic.h:102
[H5VL_dataset_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372)
H5VL_dataset_get_t
**Definition** H5VLconnector.h:218
[H5VL_DATASET_GET_DAPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a04105bc0c224fc85120a6794b1141211)
@ H5VL_DATASET_GET_DAPL
**Definition** H5VLconnector.h:219
[H5VL_DATASET_GET_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a31d5a7839c3c4ece749c4d4b5e5f2ea9)
@ H5VL_DATASET_GET_SPACE
**Definition** H5VLconnector.h:221
[H5VL_DATASET_GET_DCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a5c35e81f46b4a93c0efc89e2a0b97550)
@ H5VL_DATASET_GET_DCPL
**Definition** H5VLconnector.h:220
[H5VL_DATASET_GET_STORAGE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a64752afc213ac4e5fe4d954b78442adc)
@ H5VL_DATASET_GET_STORAGE_SIZE
**Definition** H5VLconnector.h:223
[H5VL_DATASET_GET_SPACE_STATUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372a7514a11451c7d583ac8a247e36e62906)
@ H5VL_DATASET_GET_SPACE_STATUS
**Definition** H5VLconnector.h:222
[H5VL_DATASET_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab4c18c2151788eaae8f602f4f1b9c372aa83d7cc17db73e5eb310687f3e2d3a9f)
@ H5VL_DATASET_GET_TYPE
**Definition** H5VLconnector.h:224
[H5VL_dataset_get_args_t::space_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a0327c24b4b50fe1c105339cb7ce599ab)
hid_t space_id
**Definition** H5VLconnector.h:245
[H5VL_dataset_get_args_t::dcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a3b10a61cf12d9536d1d91ae50313a5da)
hid_t dcpl_id
**Definition** H5VLconnector.h:240
[H5VL_dataset_get_args_t::dapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a6ee841fff797bec3dab658a3e113b747)
hid_t dapl_id
**Definition** H5VLconnector.h:235
[H5VL_dataset_get_args_t::type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac)
hid_t type_id
**Definition** H5VLconnector.h:260
[H5VL_dataset_get_args_t::get_space](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#aa0c6e1eb6a730ce87ef4aeb5c2946f5b)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@142367040332177125165115141324134254346032133036 get_space
[H5VL_dataset_get_args_t::storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#aa1b9b69e3b88a073d07acbd7a2a96f31)
hsize_t * storage_size
**Definition** H5VLconnector.h:255
[H5VL_dataset_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ab4ffab5d9944ce8debc82cfd8d15953c)
union H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372 args
[H5VL_dataset_get_args_t::get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ac28d98e8c75a9d519ab6d0526ca8a548)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@215223264265060230271205035143324317305335164164 get_type
[H5VL_dataset_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#acc2d43af01dd0431426e99cb3b7058d2)
H5VL_dataset_get_t op_type
**Definition** H5VLconnector.h:229
[H5VL_dataset_get_args_t::get_space_status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ad09bb951f5850c3e8381f2c34ac43214)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@150161063041315312027236166101260373221011343125 get_space_status
[H5VL_dataset_get_args_t::get_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#ad9772107bfc2eaced2ccbfd8f8f7f968)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@375173337001243367224131244063251014115163115356 get_storage_size
[H5VL_dataset_get_args_t::status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#add2a21e9cdd0ea1b51f4956858b999d5)
H5D_space_status_t * status
**Definition** H5VLconnector.h:250
[H5VL_dataset_get_args_t::get_dcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#af400cdfe9e1a85b8ac50828ad9c5c8f4)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@052016134364123326013203016121105255350053011171 get_dcpl
[H5VL_dataset_get_args_t::get_dapl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html#af8c5b98e5690e7e7db06f29f6593b8e9)
struct H5VL_dataset_get_args_t::@153335320133131360224107013164003240247000113372::@172060242312276330050052122045006035177105204004 get_dapl
###  dataset: specific
The _specific_ callback in the dataset class implements specific operations on HDF5 datasets as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req); [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) **Definition** H5VLconnector.h:457  
Arguments:   
obj (IN): The dset where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for dataset 'specific' operation */
typedef enum [H5VL_dataset_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eb) {
[H5VL_DATASET_SET_EXTENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346ebab34a36df015aa4e63a8da5312cef2b3f), /* H5Dset_extent */
[H5VL_DATASET_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eba3a67be065b4404a8521154b7a1d936db), /* H5Dflush */
[H5VL_DATASET_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346ebace7df8a591dfcc28ef96d875744ce038) /* H5Drefresh */
} [H5VL_dataset_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eb);
/* Parameters for dataset 'specific' operations */
typedef struct [H5VL_dataset_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html) {
[H5VL_dataset_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eb) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a4c838af488c33410abd5447c5bf69e83); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_DATASET_SET_EXTENT */
struct {
const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *[size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a152917c01cb0cf9c11d4373419f6e172); /* New dataspace extent */
} [set_extent](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#aa04b8385940fc26b3bfe1966a79eda51);
/* H5VL_DATASET_FLUSH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [dset_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#ac03a0858609eaef6a8186380453e4867); /* Dataset ID (IN) */
} [flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a1f7b08ec3c24c0b77347cffe90f0959c);
/* H5VL_DATASET_REFRESH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [dset_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#ac03a0858609eaef6a8186380453e4867); /* Dataset ID (IN) */
} [refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#af083586502787c54bdc78d1db20e16aa);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a2109363d4758637a57b9cb066c2a0a92);
} [H5VL_dataset_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html);
[H5VL_dataset_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eb)
H5VL_dataset_specific_t
**Definition** H5VLconnector.h:266
[H5VL_DATASET_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346eba3a67be065b4404a8521154b7a1d936db)
@ H5VL_DATASET_FLUSH
**Definition** H5VLconnector.h:268
[H5VL_DATASET_SET_EXTENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346ebab34a36df015aa4e63a8da5312cef2b3f)
@ H5VL_DATASET_SET_EXTENT
**Definition** H5VLconnector.h:267
[H5VL_DATASET_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8fefe4297e306615172689a42d2346ebace7df8a591dfcc28ef96d875744ce038)
@ H5VL_DATASET_REFRESH
**Definition** H5VLconnector.h:269
[H5VL_dataset_specific_args_t::size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a152917c01cb0cf9c11d4373419f6e172)
const hsize_t * size
**Definition** H5VLconnector.h:280
[H5VL_dataset_specific_args_t::flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a1f7b08ec3c24c0b77347cffe90f0959c)
struct H5VL_dataset_specific_args_t::@150243122340346021165032204017122066350337114043::@313222234057102123160361125346075030344144265243 flush
[H5VL_dataset_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a2109363d4758637a57b9cb066c2a0a92)
union H5VL_dataset_specific_args_t::@150243122340346021165032204017122066350337114043 args
[H5VL_dataset_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#a4c838af488c33410abd5447c5bf69e83)
H5VL_dataset_specific_t op_type
**Definition** H5VLconnector.h:274
[H5VL_dataset_specific_args_t::set_extent](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#aa04b8385940fc26b3bfe1966a79eda51)
struct H5VL_dataset_specific_args_t::@150243122340346021165032204017122066350337114043::@254004050223001216016020361033237354012264007113 set_extent
[H5VL_dataset_specific_args_t::dset_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#ac03a0858609eaef6a8186380453e4867)
hid_t dset_id
**Definition** H5VLconnector.h:285
[H5VL_dataset_specific_args_t::refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html#af083586502787c54bdc78d1db20e16aa)
struct H5VL_dataset_specific_args_t::@150243122340346021165032204017122066350337114043::@327311204244102012344152307013001036032122233214 refresh
###  dataset: optional
The _optional_ callback in the dataset class implements connector specific operations on an HDF5 dataset. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  dataset: close
The _close_ callback in the dataset class terminates access to the dataset object and free all resources it was consuming and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close)(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
dset (IN): Pointer to the dataset object. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
##  Datatype Callbacks
The HDF5 datatype routines ([Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)) allow users to create and manage HDF5 datatypes. Those routines are divided into two categories. One that operates on all types of datatypes but do not modify the contents of the container (all in memory), and others that operate on named datatypes by accessing the container. When a user creates an HDF5 datatype, it is still an object in memory space (transient datatype) that has not been added to the HDF5 containers. Only when a user commits the HDF5 datatype, it becomes persistent in the container. Those are called named/committed datatypes. The transient H5T routines should work on named datatypes nevertheless.
All the [Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) API routines that modify the HDF5 container map to one of the named datatype callback routines in this class that the connector needs to implement.
_Structure for datatype callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_datatype_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html) {
void *(*commit)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
void *(*open)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#afde5bd0d12e9ea0538088342292edbab))(void *obj, [H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#a25436a00c294004a9f1066696cd4781e))(void *obj, [H5VL_datatype_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#aea0bf6576dffcb8806350946f6a025e8))(void *dt, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_datatype_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html);
[H5VL_datatype_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html)
**Definition** H5VLconnector.h:902
[H5VL_datatype_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#a25436a00c294004a9f1066696cd4781e)
herr_t(* specific)(void *obj, H5VL_datatype_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:908
[H5VL_datatype_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#a533cec569e2019874f02dd9552bcb68e)
herr_t(* optional)(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:909
[H5VL_datatype_class_t::close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#aea0bf6576dffcb8806350946f6a025e8)
herr_t(* close)(void *dt, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:910
[H5VL_datatype_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__class__t.html#afde5bd0d12e9ea0538088342292edbab)
herr_t(* get)(void *obj, H5VL_datatype_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:907
[H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html)
**Definition** H5VLconnector.h:306
[H5VL_datatype_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html)
**Definition** H5VLconnector.h:336
###  datatype: commit
The _commit_ callback in the named datatype class creates a datatype object in the container of the location object and returns a pointer to the datatype structure containing information to access the datatype in future calls. 
Signature:   
---  
void *(*commit)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the datatype needs to be committed or where the look-up of the target object needs to start. loc_params (IN): Pointer to location parameters as explained in "Mapping the API to the Callbacks". In this call, the location type is always [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e). name (IN): The name of the datatype to be created. typeid (IN): The transient datatype identifier to be committed. lcpl_id (IN): The link creation property list. tcpl_id (IN): The datatype creation property list. tapl_id (IN): The datatype access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  datatype: open
The _open_ callback in the named datatype class opens a previously committed datatype object in the container of the location object and returns a pointer to the datatype structure containing information to access the datatype in future calls. 
Signature:   
---  
void *(*open) (void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char * name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the datatype needs to be opened or where the look-up of the target object needs to start. loc_params (IN): Pointer to location parameters as explained in "Mapping the API to the Callbacks". In this call, the location type is always [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e). name (IN): The name of the datatype to be opened. tapl_id (IN): The datatype access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  datatype: get
The _get_ callback in the named datatype class retrieves information about the named datatype as specified in thegettypeparameter.It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get) (void *obj, [H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The named datatype to retrieve information from. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for datatype 'get' operation */
typedef enum [H5VL_datatype_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045) {
[H5VL_DATATYPE_GET_BINARY_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045ac6262bbc02f5f314dd61433ac8eb1ac7), /* Get size of serialized form of transient type */
[H5VL_DATATYPE_GET_BINARY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045a7a7beb42dec3775754d11081d8393b3f), /* Get serialized form of transient type */
[H5VL_DATATYPE_GET_TCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045acae4838a97a29e1530f208320065fc4d) /* Datatype creation property list */
} [H5VL_datatype_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045);
/* Parameters for datatype 'get' operations */
typedef struct [H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html) {
[H5VL_datatype_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a4c40da96f9e63b9f4a8b9ad0d6b90fd6); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_DATATYPE_GET_BINARY_SIZE */
struct {
size_t *[size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#ae3c15c00c8f7ae14ae18501067cb5ac4); /* Size of serialized form of datatype (OUT) */
} [get_binary_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a2bb1969a4ed0c0280512945a9f7e5618);
/* H5VL_DATATYPE_GET_BINARY */
struct {
void *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a5bc5fa69bee375df074734a2c4858604); /* Buffer to store serialized form of datatype (OUT) */
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of serialized datatype buffer */
} [get_binary](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a09c15db20c9e3cb9560d6af18947b598);
/* H5VL_DATATYPE_GET_TCPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [tcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#aff53ff9d15547e7a8e31a4ff39295349); /* Named datatype creation property list ID (OUT) */
} [get_tcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#aa4dd54f1472ab05e1bbc11391f000d3c);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a2728b46562a1ae8f6abc688de11ecd00);
} [H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html);
[H5VL_datatype_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045)
H5VL_datatype_get_t
**Definition** H5VLconnector.h:299
[H5VL_DATATYPE_GET_BINARY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045a7a7beb42dec3775754d11081d8393b3f)
@ H5VL_DATATYPE_GET_BINARY
**Definition** H5VLconnector.h:301
[H5VL_DATATYPE_GET_BINARY_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045ac6262bbc02f5f314dd61433ac8eb1ac7)
@ H5VL_DATATYPE_GET_BINARY_SIZE
**Definition** H5VLconnector.h:300
[H5VL_DATATYPE_GET_TCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a50c0522b08c0dcb658b82f089b82f045acae4838a97a29e1530f208320065fc4d)
@ H5VL_DATATYPE_GET_TCPL
**Definition** H5VLconnector.h:302
[H5VL_datatype_get_args_t::get_binary](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a09c15db20c9e3cb9560d6af18947b598)
struct H5VL_datatype_get_args_t::@207210347060164211074314155130167207266250377233::@152302211012210131235064104317035100077231317006 get_binary
[H5VL_datatype_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a2728b46562a1ae8f6abc688de11ecd00)
union H5VL_datatype_get_args_t::@207210347060164211074314155130167207266250377233 args
[H5VL_datatype_get_args_t::get_binary_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a2bb1969a4ed0c0280512945a9f7e5618)
struct H5VL_datatype_get_args_t::@207210347060164211074314155130167207266250377233::@115001002205056166276243270107354227350303360136 get_binary_size
[H5VL_datatype_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a4c40da96f9e63b9f4a8b9ad0d6b90fd6)
H5VL_datatype_get_t op_type
**Definition** H5VLconnector.h:307
[H5VL_datatype_get_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#a5bc5fa69bee375df074734a2c4858604)
void * buf
**Definition** H5VLconnector.h:318
[H5VL_datatype_get_args_t::get_tcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#aa4dd54f1472ab05e1bbc11391f000d3c)
struct H5VL_datatype_get_args_t::@207210347060164211074314155130167207266250377233::@025011226023117242373074017054171120314254313273 get_tcpl
[H5VL_datatype_get_args_t::size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#ae3c15c00c8f7ae14ae18501067cb5ac4)
size_t * size
**Definition** H5VLconnector.h:313
[H5VL_datatype_get_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:319
[H5VL_datatype_get_args_t::tcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html#aff53ff9d15547e7a8e31a4ff39295349)
hid_t tcpl_id
**Definition** H5VLconnector.h:324
###  datatype: specific
The _specific_ callback in the datatype class implements specific operations on HDF5 named datatypes as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req); [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) **Definition** H5VLconnector.h:734  
Arguments:   
obj (IN): The container or object where the operation needs to happen. loc_params (IN): Pointer to location parameters as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for datatype 'specific' operation */
typedef enum [H5VL_datatype_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581) {
[H5VL_DATATYPE_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581a31de802d1231e3f005667efdbd9e849b), /* H5Tflush */
[H5VL_DATATYPE_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581a789ab10e1a6d5b33b7fbe769ee97bdc1) /* H5Trefresh */
} [H5VL_datatype_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581);
/* Parameters for datatype 'specific' operations */
typedef struct [H5VL_datatype_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html) {
[H5VL_datatype_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#aed7261f32ee4fa0b6e71e00f56f3a005); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_DATATYPE_FLUSH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac); /* Named datatype ID (IN) */
} [flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a027cbfb166e74ff6fe0741bcb17cf38c);
/* H5VL_DATATYPE_REFRESH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac); /* Named datatype ID (IN) */
} [refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a7f5ea2006475b4ef1262223844ea684d);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#af42d7a073ea8fd4cdad7ca46c72649d4);
} [H5VL_datatype_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html);
[H5VL_datatype_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581)
H5VL_datatype_specific_t
**Definition** H5VLconnector.h:330
[H5VL_DATATYPE_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581a31de802d1231e3f005667efdbd9e849b)
@ H5VL_DATATYPE_FLUSH
**Definition** H5VLconnector.h:331
[H5VL_DATATYPE_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a3717bbb5258f48e253ff646f4255b581a789ab10e1a6d5b33b7fbe769ee97bdc1)
@ H5VL_DATATYPE_REFRESH
**Definition** H5VLconnector.h:332
[H5VL_datatype_specific_args_t::flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a027cbfb166e74ff6fe0741bcb17cf38c)
struct H5VL_datatype_specific_args_t::@361346243207036156133373007120171322013226133141::@054333252246127145325026051037007256166006037342 flush
[H5VL_datatype_specific_args_t::refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a7f5ea2006475b4ef1262223844ea684d)
struct H5VL_datatype_specific_args_t::@361346243207036156133373007120171322013226133141::@042307067256103152134045014300035037305036157303 refresh
[H5VL_datatype_specific_args_t::type_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#a89fbb59cdf681f79816c5cab69c5dbac)
hid_t type_id
**Definition** H5VLconnector.h:343
[H5VL_datatype_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#aed7261f32ee4fa0b6e71e00f56f3a005)
H5VL_datatype_specific_t op_type
**Definition** H5VLconnector.h:337
[H5VL_datatype_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html#af42d7a073ea8fd4cdad7ca46c72649d4)
union H5VL_datatype_specific_args_t::@361346243207036156133373007120171322013226133141 args
###  datatype: optional
The _optional_ callback in the datatype class implements connector specific operations on an HDF5 datatype. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  datatype: close
The _close_ callback in the named datatype class terminates access to the datatype object and free all resources it was consuming and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close) (void *dt, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
dt (IN): Pointer to the datatype object. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
##  File Callbacks
The file API routines ([Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html)) allow HDF5 users to create and manage HDF5 containers. All the [Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html) API routines that modify the HDF5 container map to one of the file callback routines in his class that the connector needs to implement.
_File class for file API routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_file_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html) {
void *(*create)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
void *(*open)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#ae3c7ab10f6778059d88b1faa157d8330))(void *obj, [H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#ad2a8b6931916532d625283c956acf5fb))(void *obj, [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#aef7d9442ff871e92439d02c356e93813))(void *file, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_file_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html);
[H5VL_file_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html)
**Definition** H5VLconnector.h:914
[H5VL_file_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#a533cec569e2019874f02dd9552bcb68e)
herr_t(* optional)(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:920
[H5VL_file_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#ad2a8b6931916532d625283c956acf5fb)
herr_t(* specific)(void *obj, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:919
[H5VL_file_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#ae3c7ab10f6778059d88b1faa157d8330)
herr_t(* get)(void *obj, H5VL_file_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:918
[H5VL_file_class_t::close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__class__t.html#aef7d9442ff871e92439d02c356e93813)
herr_t(* close)(void *file, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:921
[H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html)
**Definition** H5VLconnector.h:395
###  file: create
The _create_ callback in the file class should create a container and returns a pointer to the file structure created by the connector containing information to access the container in future calls. 
Signature:   
---  
void *(*create)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, hid_tdxpl_id, void **req);  
Arguments:   
name (IN): The name of the container to be created. flags (IN): The creation flags of the container. fcpl_id (IN): The file creation property list. fapl_id (IN): The file access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  file: open
The _open_ callback in the file class should open a container and returns a pointer to the file structure created by the connector containing information to access the container in future calls. 
Signature:   
---  
void *(*open)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
name (IN): The name of the container to open. flags (IN): The open flags of the container. fapl_id (IN): The file access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  file: get
The _get_ callback in the file class should retrieve information about the container as specified in the _get_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, [H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where information needs to be retrieved from. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Info for H5VL_FILE_GET_CONT_INFO */
typedef struct [H5VL_file_cont_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html) {
unsigned [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a27fe8e82cf2ce359010cc1d08af2e105); /* version information (keep first) */
uint64_t [feature_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#aaccbcc3ce862dbe5af189ee0c3e345b1); /* Container feature flags */
/* (none currently defined) */
size_t [token_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a40f79c034c355480666082eb5e853285); /* Size of tokens */
size_t [blob_id_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a46853bd7d88b56b5a47092ed3ee693e8); /* Size of blob IDs */
} [H5VL_file_cont_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html);
/* Values for file 'get' operation */
typedef enum [H5VL_file_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178c) {
[H5VL_FILE_GET_CONT_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca2d6b91c14df76dcdc5377e9986a8d2c2), /* file get container info */
[H5VL_FILE_GET_FAPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178caf109918eace3f4c77307a26eb4a1361b), /* file access property list */
[H5VL_FILE_GET_FCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cad82b76eeca240549fff75668a8a07ffa), /* file creation property list */
[H5VL_FILE_GET_FILENO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca1ba678b6d0c35c09eebba9491a906a5b), /* file number */
[H5VL_FILE_GET_INTENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cae81f8cf07e88734b01bff3106d04050d), /* file intent */
[H5VL_FILE_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cafd2f1b2c691bb05a4c1efa44efcc8ce8), /* file name */
[H5VL_FILE_GET_OBJ_COUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cadec494437e2cd833c094952e0e7b1757), /* object count in file */
[H5VL_FILE_GET_OBJ_IDS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca97119fe8729739f6be9c61e4ef52d92a) /* object ids in file */
} [H5VL_file_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178c);
/* Parameters for file 'get_name' operation */
typedef struct [H5VL_file_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html) {
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#aed263436f49d4b0e9f6889d041634200); /* ID type of object pointer */
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of file name buffer (IN) */
char *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1); /* Buffer for file name (OUT) */
size_t *[file_name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#a9b27b1e1c81fee81e63db805f270ca64); /* Actual length of file name (OUT) */
} [H5VL_file_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html);
/* Parameters for file 'get_obj_ids' operation */
typedef struct [H5VL_file_get_obj_ids_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html) {
unsigned [types](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#ace3d9a8e1cec29ba4fbf44ecab4d3dd2); /* Type of objects to count */
size_t [max_objs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a5ea7cf00bccd42358f3668bf9a82fd0a); /* Size of array of object IDs */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *[oid_list](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a8cd7b9cccd7b6ac70bce954b004eabaa); /* Array of object IDs (OUT) */
size_t *[count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a32a30b3c1412ce372e776115c3825505); /* # of objects (OUT) */
} [H5VL_file_get_obj_ids_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html);
/* Parameters for file 'get' operations */
typedef struct [H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html) {
[H5VL_file_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178c) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a6713bc30bd7a9b2d9b597cc3d1359d20); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_FILE_GET_CONT_INFO */
struct {
H5VL_file_cont_info_t *[info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a4c19224661b5a5184376f04a091f8887); /* Container info (OUT) */
} [get_cont_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ad3af89dafdfb30502169f1ca59b5a7d0);
/* H5VL_FILE_GET_FAPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [fapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ace4df3bcfdb5bb309d7994bc043d4e5e); /* File access property list (OUT) */
} [get_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#afac2ab1c67be67c9b29dc59756fbc1a8);
/* H5VL_FILE_GET_FCPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [fcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#adfee2c69a82af269e6e472f9288c3441); /* File creation property list (OUT) */
} [get_fcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a666c421e5b631f0904ca67961026ece2);
/* H5VL_FILE_GET_FILENO */
struct {
unsigned long *[fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a5ef8cd3d2955adb5cd189a93c01daf03); /* File "number" (OUT) */
} [get_fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a2ea998d77b94b5ca1deb727407837129);
/* H5VL_FILE_GET_INTENT */
struct {
unsigned *[flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#afcd9ab20c1d046b60e462bf2895ae09b); /* File open/create intent flags (OUT) */
} [get_intent](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#aec914045fc19ea77d9ca0ed62b244a4c);
/* H5VL_FILE_GET_NAME */
H5VL_file_get_name_args_t [get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a36ebc67270b0fb5c4d1de185940efdd7);
/* H5VL_FILE_GET_OBJ_COUNT */
struct {
unsigned [types](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ace3d9a8e1cec29ba4fbf44ecab4d3dd2); /* Type of objects to count */
size_t *[count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a32a30b3c1412ce372e776115c3825505); /* # of objects (OUT) */
} [get_obj_count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#abe012b1a61e2b0ea004bf7ba515cc539);
/* H5VL_FILE_GET_OBJ_IDS */
H5VL_file_get_obj_ids_args_t [get_obj_ids](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a4c29f47671bd907a78c2bca4290fb6f9);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#aa9018c1d895456471d53e338cccbc023);
} [H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html);
[H5VL_file_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178c)
H5VL_file_get_t
**Definition** H5VLconnector.h:367
[H5VL_FILE_GET_FILENO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca1ba678b6d0c35c09eebba9491a906a5b)
@ H5VL_FILE_GET_FILENO
**Definition** H5VLconnector.h:371
[H5VL_FILE_GET_CONT_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca2d6b91c14df76dcdc5377e9986a8d2c2)
@ H5VL_FILE_GET_CONT_INFO
**Definition** H5VLconnector.h:368
[H5VL_FILE_GET_OBJ_IDS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178ca97119fe8729739f6be9c61e4ef52d92a)
@ H5VL_FILE_GET_OBJ_IDS
**Definition** H5VLconnector.h:375
[H5VL_FILE_GET_FCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cad82b76eeca240549fff75668a8a07ffa)
@ H5VL_FILE_GET_FCPL
**Definition** H5VLconnector.h:370
[H5VL_FILE_GET_OBJ_COUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cadec494437e2cd833c094952e0e7b1757)
@ H5VL_FILE_GET_OBJ_COUNT
**Definition** H5VLconnector.h:374
[H5VL_FILE_GET_INTENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cae81f8cf07e88734b01bff3106d04050d)
@ H5VL_FILE_GET_INTENT
**Definition** H5VLconnector.h:372
[H5VL_FILE_GET_FAPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178caf109918eace3f4c77307a26eb4a1361b)
@ H5VL_FILE_GET_FAPL
**Definition** H5VLconnector.h:369
[H5VL_FILE_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a693343d9d502ed7728883b7c53dc178cafd2f1b2c691bb05a4c1efa44efcc8ce8)
@ H5VL_FILE_GET_NAME
**Definition** H5VLconnector.h:373
[H5VL_file_cont_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html)
**Definition** H5VLconnector.h:358
[H5VL_file_cont_info_t::version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a27fe8e82cf2ce359010cc1d08af2e105)
unsigned version
**Definition** H5VLconnector.h:359
[H5VL_file_cont_info_t::token_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a40f79c034c355480666082eb5e853285)
size_t token_size
**Definition** H5VLconnector.h:362
[H5VL_file_cont_info_t::blob_id_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#a46853bd7d88b56b5a47092ed3ee693e8)
size_t blob_id_size
**Definition** H5VLconnector.h:363
[H5VL_file_cont_info_t::feature_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__cont__info__t.html#aaccbcc3ce862dbe5af189ee0c3e345b1)
uint64_t feature_flags
**Definition** H5VLconnector.h:360
[H5VL_file_get_args_t::get_fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a2ea998d77b94b5ca1deb727407837129)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@011017207361117153301026106354001156063133074155 get_fileno
[H5VL_file_get_args_t::count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a32a30b3c1412ce372e776115c3825505)
size_t * count
**Definition** H5VLconnector.h:431
[H5VL_file_get_args_t::get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a36ebc67270b0fb5c4d1de185940efdd7)
H5VL_file_get_name_args_t get_name
**Definition** H5VLconnector.h:426
[H5VL_file_get_args_t::info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a4c19224661b5a5184376f04a091f8887)
H5VL_file_cont_info_t * info
**Definition** H5VLconnector.h:402
[H5VL_file_get_args_t::get_obj_ids](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a4c29f47671bd907a78c2bca4290fb6f9)
H5VL_file_get_obj_ids_args_t get_obj_ids
**Definition** H5VLconnector.h:435
[H5VL_file_get_args_t::fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a5ef8cd3d2955adb5cd189a93c01daf03)
unsigned long * fileno
**Definition** H5VLconnector.h:417
[H5VL_file_get_args_t::get_fcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a666c421e5b631f0904ca67961026ece2)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@371366035142176104100261363351304244132121336161 get_fcpl
[H5VL_file_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#a6713bc30bd7a9b2d9b597cc3d1359d20)
H5VL_file_get_t op_type
**Definition** H5VLconnector.h:396
[H5VL_file_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#aa9018c1d895456471d53e338cccbc023)
union H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044 args
[H5VL_file_get_args_t::get_obj_count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#abe012b1a61e2b0ea004bf7ba515cc539)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@350007324313026124236322124026343335203256221017 get_obj_count
[H5VL_file_get_args_t::types](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ace3d9a8e1cec29ba4fbf44ecab4d3dd2)
unsigned types
**Definition** H5VLconnector.h:430
[H5VL_file_get_args_t::fapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ace4df3bcfdb5bb309d7994bc043d4e5e)
hid_t fapl_id
**Definition** H5VLconnector.h:407
[H5VL_file_get_args_t::get_cont_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#ad3af89dafdfb30502169f1ca59b5a7d0)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@272344366037253052051135316007205355242246152072 get_cont_info
[H5VL_file_get_args_t::fcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#adfee2c69a82af269e6e472f9288c3441)
hid_t fcpl_id
**Definition** H5VLconnector.h:412
[H5VL_file_get_args_t::get_intent](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#aec914045fc19ea77d9ca0ed62b244a4c)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@166254174334331062045171031105260156377232151045 get_intent
[H5VL_file_get_args_t::get_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#afac2ab1c67be67c9b29dc59756fbc1a8)
struct H5VL_file_get_args_t::@203377154345164170157356230200236207220160023044::@302374204377335267332041254076156061272064277307 get_fapl
[H5VL_file_get_args_t::flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html#afcd9ab20c1d046b60e462bf2895ae09b)
unsigned * flags
**Definition** H5VLconnector.h:422
[H5VL_file_get_name_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html)
**Definition** H5VLconnector.h:379
[H5VL_file_get_name_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1)
char * buf
**Definition** H5VLconnector.h:382
[H5VL_file_get_name_args_t::file_name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#a9b27b1e1c81fee81e63db805f270ca64)
size_t * file_name_len
**Definition** H5VLconnector.h:383
[H5VL_file_get_name_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:381
[H5VL_file_get_name_args_t::type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__name__args__t.html#aed263436f49d4b0e9f6889d041634200)
H5I_type_t type
**Definition** H5VLconnector.h:380
[H5VL_file_get_obj_ids_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html)
**Definition** H5VLconnector.h:387
[H5VL_file_get_obj_ids_args_t::count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a32a30b3c1412ce372e776115c3825505)
size_t * count
**Definition** H5VLconnector.h:391
[H5VL_file_get_obj_ids_args_t::max_objs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a5ea7cf00bccd42358f3668bf9a82fd0a)
size_t max_objs
**Definition** H5VLconnector.h:389
[H5VL_file_get_obj_ids_args_t::oid_list](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#a8cd7b9cccd7b6ac70bce954b004eabaa)
hid_t * oid_list
**Definition** H5VLconnector.h:390
[H5VL_file_get_obj_ids_args_t::types](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__obj__ids__args__t.html#ace3d9a8e1cec29ba4fbf44ecab4d3dd2)
unsigned types
**Definition** H5VLconnector.h:388
###  file: specific
The _specific_ callback in the file class implements specific operations on HDF5 files as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for file 'specific' operation */
typedef enum [H5VL_file_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776c) {
[H5VL_FILE_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776cabb47ec2cb5403edfbb59a86045914939), /* Flush file */
[H5VL_FILE_REOPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776ca4aeac39fe8593cbf9b0a0d582e910b3f), /* Reopen the file */
[H5VL_FILE_IS_ACCESSIBLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776caae709fc8e384a5795762ebdfd19156a7), /* Check if a file is accessible */
[H5VL_FILE_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776cab0272350bc97edc2b75aaf8d00f243be), /* Delete a file */
[H5VL_FILE_IS_EQUAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776ca7302e73192f78efa10ac83ca91de50b3) /* Check if two files are the same */
} [H5VL_file_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776c);
/* Parameters for file 'specific' operations */
typedef struct [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) {
[H5VL_file_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776c) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a40b4ab4d783c540283bacde5b875248c); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_FILE_FLUSH */
struct {
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) [obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a81030798b196a9008fdf5b6021205b20); /* Type of object to use */
[H5F_scope_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ac9db1b1211555797021daed9b54b8cdf) [scope](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a08d0f6a39d80f3bf72e02eb4f412b98c); /* Scope of flush operation */
} [flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a1c1a2c7e76c578caac82a973afec15e2);
/* H5VL_FILE_REOPEN */
struct {
void **[file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a763d005405f708ac1e07281969678423); /* File object for new file (OUT) */
} [reopen](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#aab5569b91fa620f4865f583e2245087c);
/* H5VL_FILE_IS_ACCESSIBLE */
struct {
const char *[filename](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a7efa5e9c7494c7d4586359300221aa5d); /* Name of file to check */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [fapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ace4df3bcfdb5bb309d7994bc043d4e5e); /* File access property list to use */
bool *[accessible](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ab377974a330f8a32670b64eca53bcd80); /* Whether file is accessible with FAPL settings (OUT) */
} [is_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a3bf01a372fa107e2e2fc9b1662879695);
/* H5VL_FILE_DELETE */
struct {
const char *[filename](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a7efa5e9c7494c7d4586359300221aa5d); /* Name of file to delete */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [fapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ace4df3bcfdb5bb309d7994bc043d4e5e); /* File access property list to use */
} [del](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a343d3c05ad7c7ea07532a0251221b6b3);
/* H5VL_FILE_IS_EQUAL */
struct {
void *[obj2](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ac2e6d717320d1b0aa4cc910db54a84e1); /* Second file object to compare against */
bool *[same_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a93490219313b1226d213c03bc29a842e); /* Whether files are the same (OUT) */
} [is_equal](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a9ef4aec5ec2f86afdac0952796695fa7);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a1210fd738bd77385aa36362dd215b836);
} [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html);
[H5F_scope_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ac9db1b1211555797021daed9b54b8cdf)
H5F_scope_t
**Definition** H5Fpublic.h:98
[H5VL_file_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776c)
H5VL_file_specific_t
**Definition** H5VLconnector.h:440
[H5VL_FILE_REOPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776ca4aeac39fe8593cbf9b0a0d582e910b3f)
@ H5VL_FILE_REOPEN
**Definition** H5VLconnector.h:442
[H5VL_FILE_IS_EQUAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776ca7302e73192f78efa10ac83ca91de50b3)
@ H5VL_FILE_IS_EQUAL
**Definition** H5VLconnector.h:453
[H5VL_FILE_IS_ACCESSIBLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776caae709fc8e384a5795762ebdfd19156a7)
@ H5VL_FILE_IS_ACCESSIBLE
**Definition** H5VLconnector.h:443
[H5VL_FILE_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776cab0272350bc97edc2b75aaf8d00f243be)
@ H5VL_FILE_DELETE
**Definition** H5VLconnector.h:452
[H5VL_FILE_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a0677561e907e85a084c9d7eb8463776cabb47ec2cb5403edfbb59a86045914939)
@ H5VL_FILE_FLUSH
**Definition** H5VLconnector.h:441
[H5VL_file_specific_args_t::scope](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a08d0f6a39d80f3bf72e02eb4f412b98c)
H5F_scope_t scope
**Definition** H5VLconnector.h:465
[H5VL_file_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a1210fd738bd77385aa36362dd215b836)
union H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172 args
[H5VL_file_specific_args_t::flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a1c1a2c7e76c578caac82a973afec15e2)
struct H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172::@327322200242155116134336214002110102277040035111 flush
[H5VL_file_specific_args_t::del](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a343d3c05ad7c7ea07532a0251221b6b3)
struct H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172::@237032256227232337104261147027366256142037327120 del
[H5VL_file_specific_args_t::is_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a3bf01a372fa107e2e2fc9b1662879695)
struct H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172::@255310331101015242163270154206156067274205204371 is_accessible
[H5VL_file_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a40b4ab4d783c540283bacde5b875248c)
H5VL_file_specific_t op_type
**Definition** H5VLconnector.h:458
[H5VL_file_specific_args_t::file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a763d005405f708ac1e07281969678423)
void ** file
**Definition** H5VLconnector.h:470
[H5VL_file_specific_args_t::filename](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a7efa5e9c7494c7d4586359300221aa5d)
const char * filename
**Definition** H5VLconnector.h:475
[H5VL_file_specific_args_t::obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a81030798b196a9008fdf5b6021205b20)
H5I_type_t obj_type
**Definition** H5VLconnector.h:464
[H5VL_file_specific_args_t::same_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a93490219313b1226d213c03bc29a842e)
bool * same_file
**Definition** H5VLconnector.h:489
[H5VL_file_specific_args_t::is_equal](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#a9ef4aec5ec2f86afdac0952796695fa7)
struct H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172::@117306305054203360013355225117110173067160012360 is_equal
[H5VL_file_specific_args_t::reopen](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#aab5569b91fa620f4865f583e2245087c)
struct H5VL_file_specific_args_t::@350160162004057176146137034057166155372175301172::@212364236151240367366272372004105226027164363233 reopen
[H5VL_file_specific_args_t::accessible](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ab377974a330f8a32670b64eca53bcd80)
bool * accessible
**Definition** H5VLconnector.h:477
[H5VL_file_specific_args_t::obj2](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ac2e6d717320d1b0aa4cc910db54a84e1)
void * obj2
**Definition** H5VLconnector.h:488
[H5VL_file_specific_args_t::fapl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html#ace4df3bcfdb5bb309d7994bc043d4e5e)
hid_t fapl_id
**Definition** H5VLconnector.h:476
###  file: optional
The _optional_ callback in the file class implements connector specific operations on an HDF5 container. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  file: close
The _close_ callback in the file class should terminate access to the file object and free all resources it was consuming, and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close)(void *file, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
file (IN): Pointer to the file. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
##  Group Callbacks
The group API routines ([Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html)) allow HDF5 users to create and manage HDF5 groups. All the [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) API routines that modify the HDF5 container map to one of the group callback routines in this class that the connector needs to implement.
_Structure for group callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_group_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html) {
void *(*create)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
void *(*open)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a6bf29d69e71a9c4ddfa9db2f93c7bb6f))(void *obj, [H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a6ec368486c41afebad682ca7830bf5f6))(void *obj, [H5VL_group_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a533cec569e2019874f02dd9552bcb68e))(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#ac5564377c35a9a35da188e9af2fe1ec9))(void *grp, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_group_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html);
[H5VL_group_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html)
**Definition** H5VLconnector.h:925
[H5VL_group_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a533cec569e2019874f02dd9552bcb68e)
herr_t(* optional)(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:932
[H5VL_group_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a6bf29d69e71a9c4ddfa9db2f93c7bb6f)
herr_t(* get)(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:930
[H5VL_group_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#a6ec368486c41afebad682ca7830bf5f6)
herr_t(* specific)(void *obj, H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:931
[H5VL_group_class_t::close](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__class__t.html#ac5564377c35a9a35da188e9af2fe1ec9)
herr_t(* close)(void *grp, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:933
[H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html)
**Definition** H5VLconnector.h:510
[H5VL_group_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html)
**Definition** H5VLconnector.h:541
###  group: create
The _create_ callback in the group class creates a group object in the container of the location object and returns a pointer to the group structure containing information to access the group in future calls. 
Signature:   
---  
void *(*create)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gcpl_id,[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the group needs to be created or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. name (IN): The name of the group to be created. dcpl_id (IN): The group creation property list. It contains all the group creation properties in addition to the link creation property list of the create operation (an [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) that can be retrieved with the property H5VL_GRP_LCPL_ID. gapl_id (IN): The group access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  group: open
The _open_ callback in the group class opens a group object in the container of the location object and returns a pointer to the group structure containing information to access the group in future calls. 
Signature:   
---  
void *(*open)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): Pointer to an object where the group needs to be opened or where the look-up of the target object needs to start. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. name (IN): The name of the group to be opened. gapl_id (IN): The group access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  group: get
The _get_ callback in the group class retrieves information about the group as specified in the _get_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, [H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req)  
Arguments:   
obj (IN): The group object where information needs to be retrieved from. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for group 'get' operation */
typedef enum [H5VL_group_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdf) {
[H5VL_GROUP_GET_GCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdfae39c8b6a3da705544c0c2d71cf89d1de), /* group creation property list */
[H5VL_GROUP_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdfa3b16d763cbaf30054b1eccfe4b18738e) /* group info */
} [H5VL_group_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdf);
/* Parameters for group 'get_info' operation */
typedef struct [H5VL_group_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html) {
[H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) [loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad); /* Location parameters for object access */
[H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) *[ginfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html#a7201e515498e21b8b16300b02a868597); /* Group info (OUT) */
} [H5VL_group_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html);
/* Parameters for group 'get' operations */
typedef struct [H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html) {
[H5VL_group_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdf) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a10ca985fc59599e9bf24d86c6a40ca39); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_GROUP_GET_GCPL */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [gcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#adaa50b67271240ed208c995eb2407245); /* Group creation property list (OUT) */
} [get_gcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a2b6792ebeb9b51a8dc1316b42be50d95);
/* H5VL_GROUP_GET_INFO */
H5VL_group_get_info_args_t [get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a578d5d1bb73d5528b90c0fbd8e7a5797); /* Group info */
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#af11e49c4df2041c647192e1c883aca8e);
} [H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html);
[H5VL_group_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdf)
H5VL_group_get_t
**Definition** H5VLconnector.h:498
[H5VL_GROUP_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdfa3b16d763cbaf30054b1eccfe4b18738e)
@ H5VL_GROUP_GET_INFO
**Definition** H5VLconnector.h:500
[H5VL_GROUP_GET_GCPL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a65f430c05c745b39b8f9664780bd1cdfae39c8b6a3da705544c0c2d71cf89d1de)
@ H5VL_GROUP_GET_GCPL
**Definition** H5VLconnector.h:499
[H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html)
**Definition** H5Gpublic.h:55
[H5VL_group_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a10ca985fc59599e9bf24d86c6a40ca39)
H5VL_group_get_t op_type
**Definition** H5VLconnector.h:511
[H5VL_group_get_args_t::get_gcpl](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a2b6792ebeb9b51a8dc1316b42be50d95)
struct H5VL_group_get_args_t::@333343267324240075252033371156074076272132061272::@302235135314326045061234204343245113127276164170 get_gcpl
[H5VL_group_get_args_t::get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#a578d5d1bb73d5528b90c0fbd8e7a5797)
H5VL_group_get_info_args_t get_info
**Definition** H5VLconnector.h:521
[H5VL_group_get_args_t::gcpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#adaa50b67271240ed208c995eb2407245)
hid_t gcpl_id
**Definition** H5VLconnector.h:517
[H5VL_group_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html#af11e49c4df2041c647192e1c883aca8e)
union H5VL_group_get_args_t::@333343267324240075252033371156074076272132061272 args
[H5VL_group_get_info_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html)
**Definition** H5VLconnector.h:504
[H5VL_group_get_info_args_t::loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html#a5bf5017b0f72740c83ee50b1ee3023ad)
H5VL_loc_params_t loc_params
**Definition** H5VLconnector.h:505
[H5VL_group_get_info_args_t::ginfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__info__args__t.html#a7201e515498e21b8b16300b02a868597)
H5G_info_t * ginfo
**Definition** H5VLconnector.h:506
###  group: specific
The _specific_ callback in the group class implements specific operations on HDF5 groups as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for group 'specific' operation */
typedef enum [H5VL_group_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565) {
[H5VL_GROUP_MOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565a91fb48cd8903ad3bb8a87a71ec734102), /* Mount a file on a group */
[H5VL_GROUP_UNMOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565a9fe35ebb106d610ef36a97de8f545416), /* Unmount a file on a group */
[H5VL_GROUP_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565aadb42eba4ff1867d6cd809b640dcc667), /* H5Gflush */
[H5VL_GROUP_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565afe3c3a9224e00ec7b2dba2b79298ddcd) /* H5Grefresh */
} [H5VL_group_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565);
/* Parameters for group 'mount' operation */
typedef struct [H5VL_group_spec_mount_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html) {
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#a8f8f80d37794cde9472343e4487ba3eb); /* Name of location to mount child file */
void *[child_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#aa523d41172cd3fc7b2c9d93dd168e290); /* Pointer to child file object */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [fmpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#a606ab1869ed77d03e087837269790234); /* File mount property list to use */
} [H5VL_group_spec_mount_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html);
/* Parameters for group 'specific' operations */
typedef struct [H5VL_group_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html) {
[H5VL_group_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#aae9e2c88692475ef2b8bee6e116bb7fc); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_GROUP_MOUNT */
H5VL_group_spec_mount_args_t [mount](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#ac833b1d756df4c0920ae2e5ec5c6a742);
/* H5VL_GROUP_UNMOUNT */
struct {
const char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a8f8f80d37794cde9472343e4487ba3eb); /* Name of location to unmount child file */
} [unmount](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#adc18483897be0a1c715083f173a7b5d1);
/* H5VL_GROUP_FLUSH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [grp_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a450c9c74199b4e8e327320e0f11e1a1b); /* Group ID (IN) */
} [flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#ac826c368830cd7b4bc98daf8dd9dce67);
/* H5VL_GROUP_REFRESH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [grp_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a450c9c74199b4e8e327320e0f11e1a1b); /* Group ID (IN) */
} [refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a21c6e45483fe0384cfc9ccaeb7707c3b);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a4f30e24adeb0d416e0fc70a62893cb0c);
} [H5VL_group_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html);
[H5VL_group_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565)
H5VL_group_specific_t
**Definition** H5VLconnector.h:526
[H5VL_GROUP_MOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565a91fb48cd8903ad3bb8a87a71ec734102)
@ H5VL_GROUP_MOUNT
**Definition** H5VLconnector.h:527
[H5VL_GROUP_UNMOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565a9fe35ebb106d610ef36a97de8f545416)
@ H5VL_GROUP_UNMOUNT
**Definition** H5VLconnector.h:528
[H5VL_GROUP_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565aadb42eba4ff1867d6cd809b640dcc667)
@ H5VL_GROUP_FLUSH
**Definition** H5VLconnector.h:529
[H5VL_GROUP_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a2d6bb6bde1b8723c9813b572db9e1565afe3c3a9224e00ec7b2dba2b79298ddcd)
@ H5VL_GROUP_REFRESH
**Definition** H5VLconnector.h:530
[H5VL_group_spec_mount_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html)
**Definition** H5VLconnector.h:534
[H5VL_group_spec_mount_args_t::fmpl_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#a606ab1869ed77d03e087837269790234)
hid_t fmpl_id
**Definition** H5VLconnector.h:537
[H5VL_group_spec_mount_args_t::name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#a8f8f80d37794cde9472343e4487ba3eb)
const char * name
**Definition** H5VLconnector.h:535
[H5VL_group_spec_mount_args_t::child_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__spec__mount__args__t.html#aa523d41172cd3fc7b2c9d93dd168e290)
void * child_file
**Definition** H5VLconnector.h:536
[H5VL_group_specific_args_t::refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a21c6e45483fe0384cfc9ccaeb7707c3b)
struct H5VL_group_specific_args_t::@026176172063377047355202013252036160164104345365::@206021050356213057175104277370273374135141122033 refresh
[H5VL_group_specific_args_t::grp_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a450c9c74199b4e8e327320e0f11e1a1b)
hid_t grp_id
**Definition** H5VLconnector.h:556
[H5VL_group_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a4f30e24adeb0d416e0fc70a62893cb0c)
union H5VL_group_specific_args_t::@026176172063377047355202013252036160164104345365 args
[H5VL_group_specific_args_t::name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#a8f8f80d37794cde9472343e4487ba3eb)
const char * name
**Definition** H5VLconnector.h:551
[H5VL_group_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#aae9e2c88692475ef2b8bee6e116bb7fc)
H5VL_group_specific_t op_type
**Definition** H5VLconnector.h:542
[H5VL_group_specific_args_t::flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#ac826c368830cd7b4bc98daf8dd9dce67)
struct H5VL_group_specific_args_t::@026176172063377047355202013252036160164104345365::@007167312366012020355171320053344247057270134032 flush
[H5VL_group_specific_args_t::mount](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#ac833b1d756df4c0920ae2e5ec5c6a742)
H5VL_group_spec_mount_args_t mount
**Definition** H5VLconnector.h:547
[H5VL_group_specific_args_t::unmount](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html#adc18483897be0a1c715083f173a7b5d1)
struct H5VL_group_specific_args_t::@026176172063377047355202013252036160164104345365::@243165323064006060342017315063115262024255075061 unmount
###  group: optional
The _optional_ callback in the group class implements connector specific operations on an HDF5 group. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where the operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  group: close
The _close_ callback in the group class terminates access to the group object and frees all resources it was consuming, and returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close)(void *group, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
group (IN): Pointer to the group object. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
##  Link Callbacks
The link API routines ([Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html)) allow HDF5 users to create and manage HDF5 links. All the [Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html) API routines that modify the HDF5 container map to one of the link callback routines in this class that the connector needs to implement.
_Structure for link callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_link_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html) {
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[create](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#a31c1efa3aa9177bc80cdf14b50f52421))([H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html) *args, void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#ab6342d2e7c8deb513e55175f64d471c5))(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj,
const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[move](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#a48572bc238a0dbbb444a5ce24cc24110))(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj,
const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#ac2629a96b496170f387a7ef1c1e142c2))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#aeb33703224992df8bddbf470b7f5d84f))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#acddb852f82f03004c66c253403b105e2))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_link_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html);
[H5VL_link_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html)
**Definition** H5VLconnector.h:937
[H5VL_link_class_t::create](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#a31c1efa3aa9177bc80cdf14b50f52421)
herr_t(* create)(H5VL_link_create_args_t *args, void *obj, const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:938
[H5VL_link_class_t::move](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#a48572bc238a0dbbb444a5ce24cc24110)
herr_t(* move)(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:943
[H5VL_link_class_t::copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#ab6342d2e7c8deb513e55175f64d471c5)
herr_t(* copy)(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:940
[H5VL_link_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#ac2629a96b496170f387a7ef1c1e142c2)
herr_t(* get)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:946
[H5VL_link_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#acddb852f82f03004c66c253403b105e2)
herr_t(* optional)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:950
[H5VL_link_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__class__t.html#aeb33703224992df8bddbf470b7f5d84f)
herr_t(* specific)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:948
[H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html)
**Definition** H5VLconnector.h:577
[H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html)
**Definition** H5VLconnector.h:610
[H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html)
**Definition** H5VLconnector.h:653
###  link: create
The _create_ callback in the group class creates a hard, soft, external, or user-defined link in the container. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*create)([H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html) *args, void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req)  
Arguments:   
args (IN/OUT): A pointer to the arguments struct. obj (IN): Pointer to an object where the link needs to be created from. loc_params (IN): Pointer to the location parameters as explained in "Mapping the API to the Callbacks" for the source object. lcplid (IN): The link creation property list. It contains all the link creation properties in addition to other API parameters depending on the creation type, which will be detailed next. laplid (IN): The link access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Link create types for VOL */
typedef enum [H5VL_link_create_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386) {
[H5VL_LINK_CREATE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386a52dc945ab2bef1f954db9d642fa5a56f),
[H5VL_LINK_CREATE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386afa2ce5d5f8fd7ef2ae93f68771211588),
[H5VL_LINK_CREATE_UD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386a4f51850fd35490934f90e300a5126f20)
} [H5VL_link_create_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386);
/* Parameters for link 'create' operations */
typedef struct [H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html) {
[H5VL_link_create_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a225b85a2fd064cdd25120b1c082c76cc); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_LINK_CREATE_HARD */
struct {
void *[curr_obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#aa5cb00f19493e7ae2dc7d21ea4124669); /* Current object */
[H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) [curr_loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a5b798d62ea87bdd52fe4f7ff94f9370f); /* Location parameters for current object */
} [hard](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#aa4f919261f4cb672762a1bbc4965f915);
/* H5VL_LINK_CREATE_SOFT */
struct {
const char *[target](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#afc76fdef3b70ccc173ac51c07b738b9d); /* Target of soft link */
} [soft](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a1268d7e81f93c6cccada085096d4bf47);
/* H5VL_LINK_CREATE_UD */
struct {
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a7c59b7f846e24f72fc4c6d21e16ed0ee); /* Type of link to create */
const void *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a0ccd7535cf45ea4a389b855324c47142); /* Buffer that contains link info */
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of link info buffer */
} [ud](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a096c678843fb1613766e44d50879c139);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a03c3d2633e618ebb4e86fad638819612);
} [H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html);
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48)
H5L_type_t
Link class types.
**Definition** H5Lpublic.h:57
[H5VL_link_create_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386)
H5VL_link_create_t
**Definition** H5VLconnector.h:570
[H5VL_LINK_CREATE_UD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386a4f51850fd35490934f90e300a5126f20)
@ H5VL_LINK_CREATE_UD
**Definition** H5VLconnector.h:573
[H5VL_LINK_CREATE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386a52dc945ab2bef1f954db9d642fa5a56f)
@ H5VL_LINK_CREATE_HARD
**Definition** H5VLconnector.h:571
[H5VL_LINK_CREATE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a8cef93e549a23b3fb2b4da12de7b0386afa2ce5d5f8fd7ef2ae93f68771211588)
@ H5VL_LINK_CREATE_SOFT
**Definition** H5VLconnector.h:572
[H5VL_link_create_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a03c3d2633e618ebb4e86fad638819612)
union H5VL_link_create_args_t::@321356074204147152175121241077171243350247053200 args
[H5VL_link_create_args_t::ud](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a096c678843fb1613766e44d50879c139)
struct H5VL_link_create_args_t::@321356074204147152175121241077171243350247053200::@314124236256374357163044356053263324076135323354 ud
[H5VL_link_create_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a0ccd7535cf45ea4a389b855324c47142)
const void * buf
**Definition** H5VLconnector.h:596
[H5VL_link_create_args_t::soft](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a1268d7e81f93c6cccada085096d4bf47)
struct H5VL_link_create_args_t::@321356074204147152175121241077171243350247053200::@251210230267352146022252041045073102105266377104 soft
[H5VL_link_create_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a225b85a2fd064cdd25120b1c082c76cc)
H5VL_link_create_t op_type
**Definition** H5VLconnector.h:578
[H5VL_link_create_args_t::curr_loc_params](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a5b798d62ea87bdd52fe4f7ff94f9370f)
H5VL_loc_params_t curr_loc_params
**Definition** H5VLconnector.h:585
[H5VL_link_create_args_t::type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#a7c59b7f846e24f72fc4c6d21e16ed0ee)
H5L_type_t type
**Definition** H5VLconnector.h:595
[H5VL_link_create_args_t::hard](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#aa4f919261f4cb672762a1bbc4965f915)
struct H5VL_link_create_args_t::@321356074204147152175121241077171243350247053200::@354300332103006322052373025226005273105320261035 hard
[H5VL_link_create_args_t::curr_obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#aa5cb00f19493e7ae2dc7d21ea4124669)
void * curr_obj
**Definition** H5VLconnector.h:584
[H5VL_link_create_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:597
[H5VL_link_create_args_t::target](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html#afc76fdef3b70ccc173ac51c07b738b9d)
const char * target
**Definition** H5VLconnector.h:590
###  link: copy
The _copy_ callback in the link class copies a link within the HDF5 container. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*copy)(void *src_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
src_obj (IN): original/source object or file. loc_params1 (IN): Pointer to the location parameters for the source object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288) in this callback. dst_obj (IN): destination object or file. loc_params2 (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288) in this callback. lcpl_id (IN): The link creation property list. lapl_id (IN): The link access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  link: move
The _move_ callback in the link class moves a link within the HDF5 container. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*move)(void *src_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
src_obj (IN): original/source object or file. loc_params1 (IN): Pointer to the location parameters for the source object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288) in this callback. dst_obj (IN): destination object or file. loc_params2 (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288) in this callback. lcpl_id (IN): The link creation property list. lapl_id (IN): The link access property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  link: get
The _get_ callback in the link class retrieves information about links as specified in the _get_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The file or group object where information needs to be retrieved from. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a256902987e95589efc0a75b709ae9288) or [H5VL_OBJECT_BY_IDX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2a4e82e176436dfcd528f38d2e76863a1e) in this callback. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for link 'get' operation */
typedef enum [H5VL_link_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5) {
[H5VL_LINK_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5adfae11f356f74a4c75efb11f18aaf024), /* link info */
[H5VL_LINK_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5a0291fdc5dfdb06f8eebb46e5abf6f9e1), /* link name */
[H5VL_LINK_GET_VAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5a0d6e014f38ca9e25747bca316c848b36) /* link value */
} [H5VL_link_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5);
/* Parameters for link 'get' operations */
typedef struct [H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html) {
[H5VL_link_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ac0f81331269da3fccb92179c11e77cab); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_LINK_GET_INFO */
struct {
[H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *[linfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a15d1f85f84b49d204d695eb516f91955); /* Pointer to link's info (OUT) */
} [get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a07f1e40724cc2df88a67a567008d63a4);
/* H5VL_LINK_GET_NAME */
struct {
size_t [name_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ab0cc47d2d65ccc71536a76101750e961); /* Size of link name buffer (IN) */
char *[name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a5ac083a645d964373f022d03df4849c8); /* Buffer for link name (OUT) */
size_t *[name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ad809338ce51cfdc277261c27b9e11619); /* Actual length of link name (OUT) */
} [get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ad9fd7db3905dd0c22f3409ed8e536660);
/* H5VL_LINK_GET_VAL */
struct {
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of link value buffer (IN) */
void *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a5bc5fa69bee375df074734a2c4858604); /* Buffer for link value (OUT) */
} [get_val](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a55ed64f92367092c43e44627fd4c32f3);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a46592633feaa7de329067f6b2716c05b);
} [H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html);
[H5VL_link_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5)
H5VL_link_get_t
**Definition** H5VLconnector.h:603
[H5VL_LINK_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5a0291fdc5dfdb06f8eebb46e5abf6f9e1)
@ H5VL_LINK_GET_NAME
**Definition** H5VLconnector.h:605
[H5VL_LINK_GET_VAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5a0d6e014f38ca9e25747bca316c848b36)
@ H5VL_LINK_GET_VAL
**Definition** H5VLconnector.h:606
[H5VL_LINK_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ad839695aa27e548b79a80416133e35e5adfae11f356f74a4c75efb11f18aaf024)
@ H5VL_LINK_GET_INFO
**Definition** H5VLconnector.h:604
[H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)
Information struct for links.
**Definition** H5Lpublic.h:76
[H5VL_link_get_args_t::get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a07f1e40724cc2df88a67a567008d63a4)
struct H5VL_link_get_args_t::@113136144362365073150253333034375224042265021277::@172130307036072251121044315101050302133164055344 get_info
[H5VL_link_get_args_t::linfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a15d1f85f84b49d204d695eb516f91955)
H5L_info2_t * linfo
**Definition** H5VLconnector.h:617
[H5VL_link_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a46592633feaa7de329067f6b2716c05b)
union H5VL_link_get_args_t::@113136144362365073150253333034375224042265021277 args
[H5VL_link_get_args_t::get_val](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a55ed64f92367092c43e44627fd4c32f3)
struct H5VL_link_get_args_t::@113136144362365073150253333034375224042265021277::@367077225027341310162131021232113246327317267260 get_val
[H5VL_link_get_args_t::name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a5ac083a645d964373f022d03df4849c8)
char * name
**Definition** H5VLconnector.h:623
[H5VL_link_get_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#a5bc5fa69bee375df074734a2c4858604)
void * buf
**Definition** H5VLconnector.h:630
[H5VL_link_get_args_t::name_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ab0cc47d2d65ccc71536a76101750e961)
size_t name_size
**Definition** H5VLconnector.h:622
[H5VL_link_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ac0f81331269da3fccb92179c11e77cab)
H5VL_link_get_t op_type
**Definition** H5VLconnector.h:611
[H5VL_link_get_args_t::name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ad809338ce51cfdc277261c27b9e11619)
size_t * name_len
**Definition** H5VLconnector.h:624
[H5VL_link_get_args_t::get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ad9fd7db3905dd0c22f3409ed8e536660)
struct H5VL_link_get_args_t::@113136144362365073150253333034375224042265021277::@016144335352156373344013262017232174155161047105 get_name
[H5VL_link_get_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:629
###  link: specific
The _specific_ callback in the link class implements specific operations on HDF5 links as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req)  
Arguments:   
obj (IN): The location object where operation needs to happen. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for link 'specific' operation */
typedef enum [H5VL_link_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60) {
[H5VL_LINK_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60ab15e04ae2e2c683fa7c54b0e11a690ef), /* H5Ldelete(_by_idx) */
[H5VL_LINK_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60a9ca6090641c2257302d39018f7c16ba8), /* link existence */
[H5VL_LINK_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60a050e4e3a3f7ecd206614cc6f533ae9af) /* H5Literate/visit(_by_name) */
} [H5VL_link_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60);
/* Parameters for link 'iterate' operation */
typedef struct [H5VL_link_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html) {
bool [recursive](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a4dfde4fec9731901f89801f08296457f); /* Whether iteration is recursive */
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) [idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0); /* Type of index to iterate over */
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) [order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb); /* Order of index iteration */
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *[idx_p](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a1cf8ef76fd3b75d2be1e39b8ce46e838); /* Start/stop iteration index (OUT) */
[H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) [op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#ac1731fba626bde475761b0a7a1780b90); /* Iteration callback function */
void *[op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#aa1649fef45b8c728cf110597e5d51f45); /* Iteration callback context */
} [H5VL_link_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html);
/* Parameters for link 'specific' operations */
typedef struct [H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html) {
[H5VL_link_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a8eeab7f9a50027e1d0216c762eaa313d); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_LINK_DELETE */
/* No args */
/* H5VL_LINK_EXISTS */
struct {
bool *[exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c); /* Whether link exists (OUT) */
} [exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c);
/* H5VL_LINK_ITER */
H5VL_link_iterate_args_t [iterate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#ac6ba2eb1ed6310ec6879ffe1ed55c558);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a7bc11e751a1e2d7d1807951d69416f44);
} [H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html);
[H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)
herr_t(* H5L_iterate2_t)(hid_t group, const char *name, const H5L_info2_t *info, void *op_data)
Prototype for H5Literate2(), H5Literate_by_name2() operator.
**Definition** H5Lpublic.h:94
[H5VL_link_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60)
H5VL_link_specific_t
**Definition** H5VLconnector.h:636
[H5VL_LINK_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60a050e4e3a3f7ecd206614cc6f533ae9af)
@ H5VL_LINK_ITER
**Definition** H5VLconnector.h:639
[H5VL_LINK_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60a9ca6090641c2257302d39018f7c16ba8)
@ H5VL_LINK_EXISTS
**Definition** H5VLconnector.h:638
[H5VL_LINK_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a03bf895394e4414592f12930030e4f60ab15e04ae2e2c683fa7c54b0e11a690ef)
@ H5VL_LINK_DELETE
**Definition** H5VLconnector.h:637
[H5VL_link_iterate_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html)
**Definition** H5VLconnector.h:643
[H5VL_link_iterate_args_t::idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0)
H5_index_t idx_type
**Definition** H5VLconnector.h:645
[H5VL_link_iterate_args_t::idx_p](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a1cf8ef76fd3b75d2be1e39b8ce46e838)
hsize_t * idx_p
**Definition** H5VLconnector.h:647
[H5VL_link_iterate_args_t::recursive](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#a4dfde4fec9731901f89801f08296457f)
bool recursive
**Definition** H5VLconnector.h:644
[H5VL_link_iterate_args_t::op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#aa1649fef45b8c728cf110597e5d51f45)
void * op_data
**Definition** H5VLconnector.h:649
[H5VL_link_iterate_args_t::order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb)
H5_iter_order_t order
**Definition** H5VLconnector.h:646
[H5VL_link_iterate_args_t::op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__iterate__args__t.html#ac1731fba626bde475761b0a7a1780b90)
H5L_iterate2_t op
**Definition** H5VLconnector.h:648
[H5VL_link_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a7bc11e751a1e2d7d1807951d69416f44)
union H5VL_link_specific_args_t::@167370167024242237262131315173253325233171101142 args
[H5VL_link_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a8eeab7f9a50027e1d0216c762eaa313d)
H5VL_link_specific_t op_type
**Definition** H5VLconnector.h:654
[H5VL_link_specific_args_t::exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c)
bool * exists
**Definition** H5VLconnector.h:663
[H5VL_link_specific_args_t::iterate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html#ac6ba2eb1ed6310ec6879ffe1ed55c558)
H5VL_link_iterate_args_t iterate
**Definition** H5VLconnector.h:667
###  link: optional
The _optional_ callback in the link class implements connector specific operations on an HDF5 link. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where operation needs to happen. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
##  Object Callbacks
The object API routines ([Objects (H5O)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html)) allow HDF5 users to manage HDF5 group, dataset, and named datatype objects. All the [Objects (H5O)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html) API routines that modify the HDF5 container map to one of the object callback routines in this class that the connector needs to implement.
_Structure for object callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_object_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html) {
void *(*open)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) *opened_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#ae4e60ba7b856841b2d49a36c19ef4631))(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, const char *src_name, void *dst_obj,
const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#a1d17d87fc70e8360290c0f9e8ac9dd19))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#a834de40803b304f7d867649fed037aa6))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#acddb852f82f03004c66c253403b105e2))(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
} [H5VL_object_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html);
[H5VL_object_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html)
**Definition** H5VLconnector.h:955
[H5VL_object_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#a1d17d87fc70e8360290c0f9e8ac9dd19)
herr_t(* get)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:961
[H5VL_object_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#a834de40803b304f7d867649fed037aa6)
herr_t(* specific)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:963
[H5VL_object_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#acddb852f82f03004c66c253403b105e2)
herr_t(* optional)(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:965
[H5VL_object_class_t::copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__class__t.html#ae4e60ba7b856841b2d49a36c19ef4631)
herr_t(* copy)(void *src_obj, const H5VL_loc_params_t *loc_params1, const char *src_name, void *dst_obj, const H5VL_loc_params_t *loc_params2, const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req)
**Definition** H5VLconnector.h:958
[H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html)
**Definition** H5VLconnector.h:684
###  object: open
The _open_ callback in the object class opens the object in the container of the location object and returns a pointer to the object structure containing information to access the object in future calls. 
Signature:   
---  
void *(*open)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) *opened_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): The container or object where operation needs to happen. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". opened_type (OUT): Buffer to return the type of the object opened ([H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6) or [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987) or [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64)). dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector. [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6) @ H5I_GROUP **Definition** H5Ipublic.h:38 [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987) @ H5I_DATASET **Definition** H5Ipublic.h:41 [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64) @ H5I_DATATYPE **Definition** H5Ipublic.h:39  
###  object: copy
The _copy_ callback in the object class copies the object from the source object to the destination object. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*copy)(void *src_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, const char *src_name, void *dst_obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req)  
Arguments:   
src_obj (IN): Pointer to location of the source object to be copied. loc_params1 (IN): Pointer to the location parameters for the source object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. src_name (IN): Name of the source object to be copied. dst_obj (IN): Pointer to location of the destination object or file. loc_params2 (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". The type can be only [H5VL_OBJECT_BY_SELF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a69fb5f76c678a35c3e7bae98c4f2c9f2ad0da87ca147a4a0507862eb8455c4b0e) in this callback. dst_name (IN): Name o be assigned to the new copy. ocpypl_id (IN): The object copy property list. lcpl_id (IN): The link creation property list. dxpl_id (IN): The data transfer property list. req (IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
###  object: get
The _get_ callback in the object class retrieves information about the object as specified in the _get_type_ parameter.It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req)  
Arguments:   
obj (IN): A location object where information needs to be retrieved from. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for object 'get' operation */
typedef enum [H5VL_object_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526e) {
[H5VL_OBJECT_GET_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eaa179e57b1c18a9fd2c5f7cb0cd138692), /* object file */
[H5VL_OBJECT_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eac0fe97360433711d6a7060cd536e6242), /* object name */
[H5VL_OBJECT_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526ea6a873ff5fcd4060f3c179552b44ebb44), /* object type */
[H5VL_OBJECT_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eaf66f04c3e08e2d2f26b794102dd94f1f) /* H5Oget_info(_by_idx|name) */
} [H5VL_object_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526e);
/* Parameters for object 'get' operations */
typedef struct [H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html) {
[H5VL_object_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526e) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#aa70fcb2e2b1a6c10b5fb4c83f011c352); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_OBJECT_GET_FILE */
struct {
void **[file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a763d005405f708ac1e07281969678423); /* File object (OUT) */
} [get_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#acd2f47d6ff525dca34304c69e348b20c);
/* H5VL_OBJECT_GET_NAME */
struct {
size_t [buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ae6563dde7454192031694b48405393d7); /* Size of name buffer (IN) */
char *[buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1); /* Buffer for name (OUT) */
size_t *[name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ad809338ce51cfdc277261c27b9e11619); /* Actual length of name (OUT) */
} [get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a37601739a165b79d2b23d461295836b8);
/* H5VL_OBJECT_GET_TYPE */
struct {
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) *[obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#af82707bd9c96143c0aeeebd4692ec91e); /* Type of object (OUT) */
} [get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ab3df60ec0df973a80333a13932bba3b6);
/* H5VL_OBJECT_GET_INFO */
struct {
unsigned [fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a807484b480420922f8646f90f5fe3bda); /* Flags for fields to retrieve */
H5O_info2_t *[oinfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a8ad8fad4efc6ec1733a37ecade29646a); /* Pointer to object info (OUT) */
} [get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a9344098b6220323a51a6e1114d1eae3c);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a6694eb5711458eb822367168f34c23e4);
} [H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html);
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec)
H5O_type_t
**Definition** H5Opublic.h:107
[H5VL_object_get_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526e)
H5VL_object_get_t
**Definition** H5VLconnector.h:676
[H5VL_OBJECT_GET_TYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526ea6a873ff5fcd4060f3c179552b44ebb44)
@ H5VL_OBJECT_GET_TYPE
**Definition** H5VLconnector.h:679
[H5VL_OBJECT_GET_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eaa179e57b1c18a9fd2c5f7cb0cd138692)
@ H5VL_OBJECT_GET_FILE
**Definition** H5VLconnector.h:677
[H5VL_OBJECT_GET_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eac0fe97360433711d6a7060cd536e6242)
@ H5VL_OBJECT_GET_NAME
**Definition** H5VLconnector.h:678
[H5VL_OBJECT_GET_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#ab745a5955edee348875cb2312c9d526eaf66f04c3e08e2d2f26b794102dd94f1f)
@ H5VL_OBJECT_GET_INFO
**Definition** H5VLconnector.h:680
[H5VL_object_get_args_t::buf](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a1fe855c208bc17a51a4d34fefdb2d5b1)
char * buf
**Definition** H5VLconnector.h:697
[H5VL_object_get_args_t::get_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a37601739a165b79d2b23d461295836b8)
struct H5VL_object_get_args_t::@352130112131156373063211320350222140001066225312::@173003014157303202055262317310241354006250362370 get_name
[H5VL_object_get_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a6694eb5711458eb822367168f34c23e4)
union H5VL_object_get_args_t::@352130112131156373063211320350222140001066225312 args
[H5VL_object_get_args_t::file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a763d005405f708ac1e07281969678423)
void ** file
**Definition** H5VLconnector.h:691
[H5VL_object_get_args_t::fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a807484b480420922f8646f90f5fe3bda)
unsigned fields
**Definition** H5VLconnector.h:708
[H5VL_object_get_args_t::oinfo](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a8ad8fad4efc6ec1733a37ecade29646a)
H5O_info2_t * oinfo
**Definition** H5VLconnector.h:709
[H5VL_object_get_args_t::get_info](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#a9344098b6220323a51a6e1114d1eae3c)
struct H5VL_object_get_args_t::@352130112131156373063211320350222140001066225312::@164260044111221266351245026126217306145002344171 get_info
[H5VL_object_get_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#aa70fcb2e2b1a6c10b5fb4c83f011c352)
H5VL_object_get_t op_type
**Definition** H5VLconnector.h:685
[H5VL_object_get_args_t::get_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ab3df60ec0df973a80333a13932bba3b6)
struct H5VL_object_get_args_t::@352130112131156373063211320350222140001066225312::@377304200060360374262244006204223067317155161020 get_type
[H5VL_object_get_args_t::get_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#acd2f47d6ff525dca34304c69e348b20c)
struct H5VL_object_get_args_t::@352130112131156373063211320350222140001066225312::@142056250136200174027260237306244055222327254074 get_file
[H5VL_object_get_args_t::name_len](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ad809338ce51cfdc277261c27b9e11619)
size_t * name_len
**Definition** H5VLconnector.h:698
[H5VL_object_get_args_t::buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#ae6563dde7454192031694b48405393d7)
size_t buf_size
**Definition** H5VLconnector.h:696
[H5VL_object_get_args_t::obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html#af82707bd9c96143c0aeeebd4692ec91e)
H5O_type_t * obj_type
**Definition** H5VLconnector.h:703
###  object: specific
The _specific_ callback in the object class implements specific operations on HDF5 objects as specified in the _specific_type_ parameter. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): A location object where he operation needs to happen. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
/* Values for object 'specific' operation */
typedef enum [H5VL_object_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867) {
[H5VL_OBJECT_CHANGE_REF_COUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a360f6e624b31a3acc685dd9718a22348), /* H5Oincr/decr_refcount */
[H5VL_OBJECT_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a028b6cfd944dffb3bc87a598e52890ec), /* H5Oexists_by_name */
[H5VL_OBJECT_LOOKUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a5f61abca48ce2e96f476894572c73250), /* Lookup object */
[H5VL_OBJECT_VISIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867ab2b90a1c6de2d3a0ea2a54a1b1143ab3), /* H5Ovisit(_by_name) */
[H5VL_OBJECT_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867acc85e8ba48c6f3af0f1c88076d5be7f5), /* H5{D|G|O|T}flush */
[H5VL_OBJECT_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867aeae649ec107e77a2e82057aac1459b8b) /* H5{D|G|O|T}refresh */
} [H5VL_object_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867);
/* Parameters for object 'visit' operation */
typedef struct [H5VL_object_visit_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html) {
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) [idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0); /* Type of index to iterate over */
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) [order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb); /* Order of index iteration */
unsigned [fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a807484b480420922f8646f90f5fe3bda); /* Flags for fields to provide in 'info' object for 'op' callback */
[H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c) [op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a3e5fd7b970d501a1bc3ff144ded1daba); /* Iteration callback function */
void *[op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#aa1649fef45b8c728cf110597e5d51f45); /* Iteration callback context */
} [H5VL_object_visit_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html);
/* Parameters for object 'specific' operations */
typedef struct [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) {
[H5VL_object_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a2b1418bc18b1dcaa0c06e2fc6bd06833); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_OBJECT_CHANGE_REF_COUNT */
struct {
int [delta](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a1dfcb70b9229f2da17dd5922b87ecf2c); /* Amount to modify object's refcount */
} [change_rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a3f5963746e7db2f96d87454e24bd65b9);
/* H5VL_OBJECT_EXISTS */
struct {
bool *[exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c); /* Whether object exists (OUT) */
} [exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c);
/* H5VL_OBJECT_LOOKUP */
struct {
H5O_token_t *[token_ptr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#acd436138e228843c3d6e9eb44621b6fd); /* Pointer to token for lookup (OUT) */
} [lookup](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a7cacb894b387935c9ab404285d34ca55);
/* H5VL_OBJECT_VISIT */
H5VL_object_visit_args_t [visit](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a02994fae7c880caf93c076e4ed930ded);
/* H5VL_OBJECT_FLUSH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [obj_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a94ecc5a5534ea8f4f1a5d5020ac1e5ce); /* Object ID (IN) */
} [flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a9d8c5aec26542aab799439d7b4e12a21);
/* H5VL_OBJECT_REFRESH */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [obj_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a94ecc5a5534ea8f4f1a5d5020ac1e5ce); /* Object ID (IN) */
} [refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a7cf1b0f882a52c806758570c4bf8d4d6);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#ac090f7d1c2082bdcc1f54ccaa485120c);
} [H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html);
[H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c)
herr_t(* H5O_iterate2_t)(hid_t obj, const char *name, const H5O_info2_t *info, void *op_data)
**Definition** H5Opublic.h:195
[H5VL_object_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867)
H5VL_object_specific_t
**Definition** H5VLconnector.h:715
[H5VL_OBJECT_EXISTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a028b6cfd944dffb3bc87a598e52890ec)
@ H5VL_OBJECT_EXISTS
**Definition** H5VLconnector.h:717
[H5VL_OBJECT_CHANGE_REF_COUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a360f6e624b31a3acc685dd9718a22348)
@ H5VL_OBJECT_CHANGE_REF_COUNT
**Definition** H5VLconnector.h:716
[H5VL_OBJECT_LOOKUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867a5f61abca48ce2e96f476894572c73250)
@ H5VL_OBJECT_LOOKUP
**Definition** H5VLconnector.h:718
[H5VL_OBJECT_VISIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867ab2b90a1c6de2d3a0ea2a54a1b1143ab3)
@ H5VL_OBJECT_VISIT
**Definition** H5VLconnector.h:719
[H5VL_OBJECT_FLUSH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867acc85e8ba48c6f3af0f1c88076d5be7f5)
@ H5VL_OBJECT_FLUSH
**Definition** H5VLconnector.h:720
[H5VL_OBJECT_REFRESH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a21596f5008ac90a1daec63fa330d5867aeae649ec107e77a2e82057aac1459b8b)
@ H5VL_OBJECT_REFRESH
**Definition** H5VLconnector.h:721
[H5VL_object_specific_args_t::visit](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a02994fae7c880caf93c076e4ed930ded)
H5VL_object_visit_args_t visit
**Definition** H5VLconnector.h:755
[H5VL_object_specific_args_t::delta](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a1dfcb70b9229f2da17dd5922b87ecf2c)
int delta
**Definition** H5VLconnector.h:741
[H5VL_object_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a2b1418bc18b1dcaa0c06e2fc6bd06833)
H5VL_object_specific_t op_type
**Definition** H5VLconnector.h:735
[H5VL_object_specific_args_t::change_rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a3f5963746e7db2f96d87454e24bd65b9)
struct H5VL_object_specific_args_t::@367057131301240221347263176354375362051322354251::@243231100274041020304122177222060150016251265254 change_rc
[H5VL_object_specific_args_t::lookup](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a7cacb894b387935c9ab404285d34ca55)
struct H5VL_object_specific_args_t::@367057131301240221347263176354375362051322354251::@132116352151033157344037206363347145151257102360 lookup
[H5VL_object_specific_args_t::refresh](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a7cf1b0f882a52c806758570c4bf8d4d6)
struct H5VL_object_specific_args_t::@367057131301240221347263176354375362051322354251::@316235325166361056043036123027027125045361321214 refresh
[H5VL_object_specific_args_t::obj_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a94ecc5a5534ea8f4f1a5d5020ac1e5ce)
hid_t obj_id
**Definition** H5VLconnector.h:759
[H5VL_object_specific_args_t::exists](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a9b2cf925137a3f578227a5fc9d4e751c)
bool * exists
**Definition** H5VLconnector.h:746
[H5VL_object_specific_args_t::flush](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#a9d8c5aec26542aab799439d7b4e12a21)
struct H5VL_object_specific_args_t::@367057131301240221347263176354375362051322354251::@323030252212253042047170143162311170344017306267 flush
[H5VL_object_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#ac090f7d1c2082bdcc1f54ccaa485120c)
union H5VL_object_specific_args_t::@367057131301240221347263176354375362051322354251 args
[H5VL_object_specific_args_t::token_ptr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html#acd436138e228843c3d6e9eb44621b6fd)
H5O_token_t * token_ptr
**Definition** H5VLconnector.h:751
[H5VL_object_visit_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html)
**Definition** H5VLconnector.h:725
[H5VL_object_visit_args_t::idx_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a15b19bb0dea4b247157e6f62850ec7a0)
H5_index_t idx_type
**Definition** H5VLconnector.h:726
[H5VL_object_visit_args_t::op](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a3e5fd7b970d501a1bc3ff144ded1daba)
H5O_iterate2_t op
**Definition** H5VLconnector.h:729
[H5VL_object_visit_args_t::fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#a807484b480420922f8646f90f5fe3bda)
unsigned fields
**Definition** H5VLconnector.h:728
[H5VL_object_visit_args_t::op_data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#aa1649fef45b8c728cf110597e5d51f45)
void * op_data
**Definition** H5VLconnector.h:730
[H5VL_object_visit_args_t::order](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__visit__args__t.html#aab445f1d17ed2d51e4ae4a268b1742bb)
H5_iter_order_t order
**Definition** H5VLconnector.h:727
###  object: optional
The _optional_ callback in the object class implements connector specific operations on an HDF5 object. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): A container or object where he operation needs to happen. loc_params (IN): Pointer to the location parameters for the destination object as explained in "Mapping the API to the Callbacks". args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
##  Introspection Callbacks
_Structure for VOL connector introspection callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_introspect_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html) {
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get_conn_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a2b1626a2b2a7ecc76c34de7b5c23017b))(void *obj, [H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21) lvl, const struct [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) **conn_cls);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a4269a6123ef519ae06305f6e95bbfca9))(const void *info, uint64_t *cap_flags);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[opt_query](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a51f75a0a76567bfd0430adb68f87101b))(void *obj, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) cls, int opt_type, uint64_t *flags);
} [H5VL_introspect_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html);
[H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21)
H5VL_get_conn_lvl_t
**Definition** H5VLconnector.h:973
[H5VL_introspect_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html)
**Definition** H5VLconnector.h:984
[H5VL_introspect_class_t::get_conn_cls](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a2b1626a2b2a7ecc76c34de7b5c23017b)
herr_t(* get_conn_cls)(void *obj, H5VL_get_conn_lvl_t lvl, const struct H5VL_class_t **conn_cls)
**Definition** H5VLconnector.h:985
[H5VL_introspect_class_t::get_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a4269a6123ef519ae06305f6e95bbfca9)
herr_t(* get_cap_flags)(const void *info, uint64_t *cap_flags)
**Definition** H5VLconnector.h:986
[H5VL_introspect_class_t::opt_query](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__introspect__class__t.html#a51f75a0a76567bfd0430adb68f87101b)
herr_t(* opt_query)(void *obj, H5VL_subclass_t cls, int opt_type, uint64_t *flags)
**Definition** H5VLconnector.h:987
###  introspect: get_conn_cls
Get a connector's [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) struct. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get_conn_cls)(void *obj, [H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21) lvl, const struct [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) **conn_cls);  
Arguments:   
obj (IN): The VOL object. lvl (IN): Current or terminal connector. cls (OUT): A const pointer to the connector.  
_The "lvl" argument is an enum:_
/* "Levels" for 'get connector class' introspection callback */
typedef enum [H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21) {
[H5VL_GET_CONN_LVL_CURR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21a8eca5ca176588a85afabb74ef23a5393), /* Get "current" connector (for this object) */
[H5VL_GET_CONN_LVL_TERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21a7849c0441ebf9d6064ac1d2ee44357e2) /* Get "terminal" connector (for this object) */
/* (Recursively called, for pass-through connectors) */
/* (Connectors that "split" must choose which connector to return) */
} [H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21);
[H5VL_GET_CONN_LVL_TERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21a7849c0441ebf9d6064ac1d2ee44357e2)
@ H5VL_GET_CONN_LVL_TERM
**Definition** H5VLconnector.h:975
[H5VL_GET_CONN_LVL_CURR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21a8eca5ca176588a85afabb74ef23a5393)
@ H5VL_GET_CONN_LVL_CURR
**Definition** H5VLconnector.h:974
###  introspect: get_cap_flags
Get a connector's capability flags. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get_cap_flags)(const void *info, unsigned *[cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#ac4b405cf0a53f734d273232a82baf247)) [H5VL_class_t::cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#ac4b405cf0a53f734d273232a82baf247) uint64_t cap_flags **Definition** H5VLconnector.h:1026  
Arguments:   
info (IN): A const pointer to pertinent VOL info. [cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html#ac4b405cf0a53f734d273232a82baf247) (OUT): A pointer to capability flags.  
###  introspect: opt_query
Query a class for a capability or functionality. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*opt_query)(void *obj, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) cls, int opt_type, bool *supported);  
Arguments:   
obj (IN): The VOL object. cls (IN): The VOL 'class' to query. opt_type (IN): The specific option to query. supported (OUT): Whether the operation is supported or not.  
_The "cls" argument is an enum:_
// Enum type for each VOL subclass
// (Used for various queries, etc)
typedef enum [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) {
[H5VL_SUBCLS_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac05e9c424f4c57ab04bb8a0f27680765), // Operations outside of a subclass
[H5VL_SUBCLS_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6382b83356def3b14a27c06488a46b62), // 'Info' subclass
[H5VL_SUBCLS_WRAP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a834b756ff01e1edb2979a0be92c3518b), // 'Wrap' subclass
[H5VL_SUBCLS_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac49cca2c43d9a93d28ded5b9dc9a14d1), // 'Attribute' subclass
[H5VL_SUBCLS_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac9b25c8d8ea5205bf8f0fb654d29a57b), // 'Dataset' subclass
[H5VL_SUBCLS_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6f3df58583ad3a02b32f1ea9e9a233c7), // 'Named datatype' subclass
[H5VL_SUBCLS_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a8611b474759b782775d303acd28c512f), // 'File' subclass
[H5VL_SUBCLS_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599aeb711151c2a908ae42e18e80bb7a8f1d), // 'Group' subclass
[H5VL_SUBCLS_LINK](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6d1475a46f9db62a48b9362362016e83), // 'Link' subclass
[H5VL_SUBCLS_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a82eb9b3d6f086cafcb446eb86534a813), // 'Object' subclass
[H5VL_SUBCLS_REQUEST](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ad23e1011cd67c7280a90ec903f210c08), // 'Request' subclass
[H5VL_SUBCLS_BLOB](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599af6ba173c8e6b99d3df13f26d4f943e66), // 'Blob' subclass
[H5VL_SUBCLS_TOKEN](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a3a4313ed710d7a5a5dbfb9ccc354c8ac) // 'Token' subclass
} [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599);
[H5VL_SUBCLS_TOKEN](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a3a4313ed710d7a5a5dbfb9ccc354c8ac)
@ H5VL_SUBCLS_TOKEN
**Definition** H5VLpublic.h:196
[H5VL_SUBCLS_INFO](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6382b83356def3b14a27c06488a46b62)
@ H5VL_SUBCLS_INFO
**Definition** H5VLpublic.h:185
[H5VL_SUBCLS_LINK](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6d1475a46f9db62a48b9362362016e83)
@ H5VL_SUBCLS_LINK
**Definition** H5VLpublic.h:192
[H5VL_SUBCLS_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a6f3df58583ad3a02b32f1ea9e9a233c7)
@ H5VL_SUBCLS_DATATYPE
**Definition** H5VLpublic.h:189
[H5VL_SUBCLS_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a82eb9b3d6f086cafcb446eb86534a813)
@ H5VL_SUBCLS_OBJECT
**Definition** H5VLpublic.h:193
[H5VL_SUBCLS_WRAP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a834b756ff01e1edb2979a0be92c3518b)
@ H5VL_SUBCLS_WRAP
**Definition** H5VLpublic.h:186
[H5VL_SUBCLS_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599a8611b474759b782775d303acd28c512f)
@ H5VL_SUBCLS_FILE
**Definition** H5VLpublic.h:190
[H5VL_SUBCLS_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac05e9c424f4c57ab04bb8a0f27680765)
@ H5VL_SUBCLS_NONE
**Definition** H5VLpublic.h:184
[H5VL_SUBCLS_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac49cca2c43d9a93d28ded5b9dc9a14d1)
@ H5VL_SUBCLS_ATTR
**Definition** H5VLpublic.h:187
[H5VL_SUBCLS_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ac9b25c8d8ea5205bf8f0fb654d29a57b)
@ H5VL_SUBCLS_DATASET
**Definition** H5VLpublic.h:188
[H5VL_SUBCLS_REQUEST](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599ad23e1011cd67c7280a90ec903f210c08)
@ H5VL_SUBCLS_REQUEST
**Definition** H5VLpublic.h:194
[H5VL_SUBCLS_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599aeb711151c2a908ae42e18e80bb7a8f1d)
@ H5VL_SUBCLS_GROUP
**Definition** H5VLpublic.h:191
[H5VL_SUBCLS_BLOB](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#gga14175ca7d867657e3e5c2f79a154a599af6ba173c8e6b99d3df13f26d4f943e66)
@ H5VL_SUBCLS_BLOB
**Definition** H5VLpublic.h:195
##  Request (Async) Callbacks
_Structure for async request callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_request_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html) {
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[wait](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a614f5a3d40d4598ea76af50389048265))(void *req, uint64_t timeout, [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[notify](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a0b3c8b20e55448d39028caf60abc1d29))(void *req, [H5VL_request_notify_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a418e1ef08bd10c57dc12f04b1e22f784) cb, void *ctx);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[cancel](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#acea4b5168303e98a0a1d0d1645847108))(void *req, [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a7532a1d5f94078199da4dd8e3003583f))(void *req, [H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html) *args);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#afe15b6d73a47c3fdcc392bf0291bf632))(void *req, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#acab3db113efe3489c4428d09e0505e30))(void *req);
} [H5VL_request_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html);
[H5VL_request_notify_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a418e1ef08bd10c57dc12f04b1e22f784)
herr_t(* H5VL_request_notify_t)(void *ctx, H5VL_request_status_t status)
**Definition** H5VLconnector.h:970
[H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b)
H5VL_request_status_t
**Definition** H5VLconnector.h:773
[H5VL_request_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html)
**Definition** H5VLconnector.h:991
[H5VL_request_class_t::notify](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a0b3c8b20e55448d39028caf60abc1d29)
herr_t(* notify)(void *req, H5VL_request_notify_t cb, void *ctx)
**Definition** H5VLconnector.h:993
[H5VL_request_class_t::wait](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a614f5a3d40d4598ea76af50389048265)
herr_t(* wait)(void *req, uint64_t timeout, H5VL_request_status_t *status)
**Definition** H5VLconnector.h:992
[H5VL_request_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#a7532a1d5f94078199da4dd8e3003583f)
herr_t(* specific)(void *req, H5VL_request_specific_args_t *args)
**Definition** H5VLconnector.h:995
[H5VL_request_class_t::free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#acab3db113efe3489c4428d09e0505e30)
herr_t(* free)(void *req)
**Definition** H5VLconnector.h:997
[H5VL_request_class_t::cancel](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#acea4b5168303e98a0a1d0d1645847108)
herr_t(* cancel)(void *req, H5VL_request_status_t *status)
**Definition** H5VLconnector.h:994
[H5VL_request_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__class__t.html#afe15b6d73a47c3fdcc392bf0291bf632)
herr_t(* optional)(void *req, H5VL_optional_args_t *args)
**Definition** H5VLconnector.h:996
[H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html)
**Definition** H5VLconnector.h:792
###  request: wait
Wait (with a timeout) for an async operation to complete. Releases the request if the operation has completed and the connector callback succeeds. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*wait)(void *req, uint64_t timeout, [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);  
Arguments:   
req (IN): The async request on which to wait. timeout (IN): The timeout value. status (IN): The status.  
_The "status" argument is an enum:_
/* Status values for async request operations */
typedef enum [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) {
[H5VL_REQUEST_STATUS_IN_PROGRESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bae94eba28d39f16c3a3fdf82032c8865b), /* Operation has not yet completed */
[H5VL_REQUEST_STATUS_SUCCEED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba73dca0b9688c46dc9b483fd0fac7fab7), /* Operation has completed, successfully */
[H5VL_REQUEST_STATUS_FAIL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bad3b05e3bbcd11850d901da87a1b6cd3a), /* Operation has completed, but failed */
[H5VL_REQUEST_STATUS_CANT_CANCEL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba77d6a4532e6296c217c54963c1281810), /* An attempt to cancel this operation was made, but it */
/* can't be canceled immediately. The operation has */
/* not completed successfully or failed, and is not yet */
/* in progress. Another attempt to cancel it may be */
/* attempted and may (or may not) succeed. */
[H5VL_REQUEST_STATUS_CANCELED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bab369014a266b97505966381bfa9b75bd) /* Operation has not completed and was canceled */
} [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b);
[H5VL_REQUEST_STATUS_SUCCEED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba73dca0b9688c46dc9b483fd0fac7fab7)
@ H5VL_REQUEST_STATUS_SUCCEED
**Definition** H5VLconnector.h:775
[H5VL_REQUEST_STATUS_CANT_CANCEL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba77d6a4532e6296c217c54963c1281810)
@ H5VL_REQUEST_STATUS_CANT_CANCEL
**Definition** H5VLconnector.h:777
[H5VL_REQUEST_STATUS_CANCELED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bab369014a266b97505966381bfa9b75bd)
@ H5VL_REQUEST_STATUS_CANCELED
**Definition** H5VLconnector.h:782
[H5VL_REQUEST_STATUS_FAIL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bad3b05e3bbcd11850d901da87a1b6cd3a)
@ H5VL_REQUEST_STATUS_FAIL
**Definition** H5VLconnector.h:776
[H5VL_REQUEST_STATUS_IN_PROGRESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bae94eba28d39f16c3a3fdf82032c8865b)
@ H5VL_REQUEST_STATUS_IN_PROGRESS
**Definition** H5VLconnector.h:774
###  request: notify
Registers a user callback to be invoked when an asynchronous operation completes. Releases the request if connector callback succeeds. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*notify)(void *req, [H5VL_request_notify_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a418e1ef08bd10c57dc12f04b1e22f784) cb, void *ctx);  
Arguments:   
req (IN): The async request that will receive the notify callback. cb (IN): The notify callback for the request. ctx (IN): The request's context.  
_The "cb" argument is an enum:_
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5VL_request_notify_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a418e1ef08bd10c57dc12f04b1e22f784))(void *ctx, [H5ES_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#aec6fb3387e2a9225fcbeaa7cf5d20365) status)
[H5ES_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#aec6fb3387e2a9225fcbeaa7cf5d20365)
H5ES_status_t
**Definition** H5ESpublic.h:53
###  request: cancel
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*cancel)(void *req, [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);  
Arguments:   
req (IN): The async request to be cancelled. status (IN): The status.  
_The "status" argument is an enum:_
/* Status values for async request operations */
typedef enum [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) {
[H5VL_REQUEST_STATUS_IN_PROGRESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bae94eba28d39f16c3a3fdf82032c8865b), /* Operation has not yet completed */
[H5VL_REQUEST_STATUS_SUCCEED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba73dca0b9688c46dc9b483fd0fac7fab7), /* Operation has completed, successfully */
[H5VL_REQUEST_STATUS_FAIL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bad3b05e3bbcd11850d901da87a1b6cd3a), /* Operation has completed, but failed */
[H5VL_REQUEST_STATUS_CANT_CANCEL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3ba77d6a4532e6296c217c54963c1281810), /* An attempt to cancel this operation was made, but it */
/* can't be canceled immediately. The operation has */
/* not completed successfully or failed, and is not yet */
/* in progress. Another attempt to cancel it may be */
/* attempted and may (or may not) succeed. */
[H5VL_REQUEST_STATUS_CANCELED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3bab369014a266b97505966381bfa9b75bd) /* Operation has not completed and was canceled */
} [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b);
###  request: specific
Perform a specific operation on an asynchronous request. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *req, [H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html) *args);  
Arguments:   
req (IN): The async request on which to perform the operation. args (IN/OUT): A pointer to the arguments struct.  
/* Values for async request 'specific' operation */
typedef enum [H5VL_request_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1) {
[H5VL_REQUEST_GET_ERR_STACK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1a68408be0a2baea1c71776339a4268f79), /* Retrieve error stack for failed operation */
[H5VL_REQUEST_GET_EXEC_TIME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1ab21c6d87c43f2b7ed354cfb8dce138cb) /* Retrieve execution time for operation */
} [H5VL_request_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1);
/* Parameters for request 'specific' operations */
typedef struct [H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html) {
[H5VL_request_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ae6ab742777db24bb7e609d486cd74e5c); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_REQUEST_GET_ERR_STACK */
struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [err_stack_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ac717c89bff4c1eac778f760deec58c31); /* Error stack ID for operation (OUT) */
} [get_err_stack](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#af555e5ecd52d15863ca2e6b8324f18e9);
/* H5VL_REQUEST_GET_EXEC_TIME */
struct {
uint64_t *[exec_ts](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a40987a5db0a269581e5a866bd0d25fb0); /* Timestamp for start of task execution (OUT) */
uint64_t *[exec_time](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ad73a499e8abec23c9de6ef4fd7acc255); /* Duration of task execution (in ns) (OUT) */
} [get_exec_time](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a50c361782a39402b63a656ed6bf67980);
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a29580fa589a9cac928ff6e53d71374f9);
} [H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html);
[H5VL_request_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1)
H5VL_request_specific_t
**Definition** H5VLconnector.h:786
[H5VL_REQUEST_GET_ERR_STACK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1a68408be0a2baea1c71776339a4268f79)
@ H5VL_REQUEST_GET_ERR_STACK
**Definition** H5VLconnector.h:787
[H5VL_REQUEST_GET_EXEC_TIME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a91890110142de649aa5674e72ba2c8c1ab21c6d87c43f2b7ed354cfb8dce138cb)
@ H5VL_REQUEST_GET_EXEC_TIME
**Definition** H5VLconnector.h:788
[H5VL_request_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a29580fa589a9cac928ff6e53d71374f9)
union H5VL_request_specific_args_t::@001316000106006326263370212102314303306103140342 args
[H5VL_request_specific_args_t::exec_ts](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a40987a5db0a269581e5a866bd0d25fb0)
uint64_t * exec_ts
**Definition** H5VLconnector.h:804
[H5VL_request_specific_args_t::get_exec_time](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#a50c361782a39402b63a656ed6bf67980)
struct H5VL_request_specific_args_t::@001316000106006326263370212102314303306103140342::@041335317214072146141141224222305140242063042067 get_exec_time
[H5VL_request_specific_args_t::err_stack_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ac717c89bff4c1eac778f760deec58c31)
hid_t err_stack_id
**Definition** H5VLconnector.h:799
[H5VL_request_specific_args_t::exec_time](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ad73a499e8abec23c9de6ef4fd7acc255)
uint64_t * exec_time
**Definition** H5VLconnector.h:805
[H5VL_request_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#ae6ab742777db24bb7e609d486cd74e5c)
H5VL_request_specific_t op_type
**Definition** H5VLconnector.h:793
[H5VL_request_specific_args_t::get_err_stack](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html#af555e5ecd52d15863ca2e6b8324f18e9)
struct H5VL_request_specific_args_t::@001316000106006326263370212102314303306103140342::@165015246060323300365012332125246340142251334111 get_err_stack
###  request: optional
Perform a connector-specific operation for a request. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *req, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);  
Arguments:   
req (IN): The async request on which to perform the operation. args (IN/OUT): A pointer to the arguments struct.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct.
###  request: free
Frees an asynchronous request. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*free)(void *req);  
Arguments:   
req (IN): The async request to be freed.  
##  Blob Callbacks
_Structure for blob callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_blob_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html) {
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[put](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#a05cc95b9cd27b93ed81195ca1136c496))(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#aa83e3900c06c07677d9454e6eab7f435))(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#a3ca3f8d0e6eecfed22112db39b942907))(void *obj, void *blob_id, [H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html) *args);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#aa1711c74af0b89880652f6f3a1c561d5))(void *obj, void *blob_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);
} [H5VL_blob_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html);
[H5VL_blob_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html)
**Definition** H5VLconnector.h:1001
[H5VL_blob_class_t::put](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#a05cc95b9cd27b93ed81195ca1136c496)
herr_t(* put)(void *obj, const void *buf, size_t size, void *blob_id, void *ctx)
**Definition** H5VLconnector.h:1002
[H5VL_blob_class_t::specific](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#a3ca3f8d0e6eecfed22112db39b942907)
herr_t(* specific)(void *obj, void *blob_id, H5VL_blob_specific_args_t *args)
**Definition** H5VLconnector.h:1004
[H5VL_blob_class_t::optional](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#aa1711c74af0b89880652f6f3a1c561d5)
herr_t(* optional)(void *obj, void *blob_id, H5VL_optional_args_t *args)
**Definition** H5VLconnector.h:1005
[H5VL_blob_class_t::get](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__class__t.html#aa83e3900c06c07677d9454e6eab7f435)
herr_t(* get)(void *obj, const void *blob_id, void *buf, size_t size, void *ctx)
**Definition** H5VLconnector.h:1003
[H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html)
**Definition** H5VLconnector.h:822
###  blob: put
Put a blob through the VOL. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*put)(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);  
Arguments:   
obj (IN): Pointer to the blob container. buf (IN): Pointer to the blob. size (IN): Size of the blob. blob_id (OUT): Pointer to the blob's connector-specific ID. ctx (IN): Connector-specific blob context.  
###  blob: get
Get a blob through the VOL. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*get)(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);  
Arguments:   
obj (IN): Pointer to the blob container. blob_id (IN): Pointer to the blob's connector-specific ID. buf (IN/OUT): Pointer to the blob. size (IN): Size of the blob. ctx (IN): Connector-specific blob context.  
###  blob: specific
Perform a defined operation on a blob via the VOL. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*specific)(void *obj, void *blob_id, [H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html) *args);  
Arguments:   
obj (IN): Pointer to the blob container. blob_id (IN): Pointer to the blob's connector-specific ID. args (IN/OUT): A pointer to the arguments struct.  
/* Values for 'blob' 'specific' operation */
typedef enum [H5VL_blob_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6) {
[H5VL_BLOB_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6ad89fddfe0d59c281d327d964cf436bb3), /* Delete a blob (by ID) */
[H5VL_BLOB_ISNULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6a07b91e3b9a440edfe61fae135016f488), /* Check if a blob ID is "null" */
[H5VL_BLOB_SETNULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6ad427093222848690bc262f2bfba8951a) /* Set a blob ID to the connector's "null" blob ID value */
} [H5VL_blob_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6);
/* Parameters for blob 'specific' operations */
typedef struct [H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html) {
[H5VL_blob_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6) [op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a78d2a3265b57217d686509b4e483d015); /* Operation to perform */
/* Parameters for each operation */
union {
/* H5VL_BLOB_DELETE */
/* No args */
/* H5VL_BLOB_ISNULL */
struct {
bool *[isnull](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a7878d87e4561a2159a80f4d4ffa157b4); /* Whether blob ID is "null" (OUT) */
} [is_null](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a9c1b136fbf6bebbfdc1c361d03904a78);
/* H5VL_BLOB_SETNULL */
/* No args */
} [args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#ac9e907954c93a08604a9de2f62e1d5f5);
} [H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html);
[H5VL_blob_specific_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6)
H5VL_blob_specific_t
**Definition** H5VLconnector.h:815
[H5VL_BLOB_ISNULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6a07b91e3b9a440edfe61fae135016f488)
@ H5VL_BLOB_ISNULL
**Definition** H5VLconnector.h:817
[H5VL_BLOB_SETNULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6ad427093222848690bc262f2bfba8951a)
@ H5VL_BLOB_SETNULL
**Definition** H5VLconnector.h:818
[H5VL_BLOB_DELETE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a62920eba2774bece9b657b94d5786bd6ad89fddfe0d59c281d327d964cf436bb3)
@ H5VL_BLOB_DELETE
**Definition** H5VLconnector.h:816
[H5VL_blob_specific_args_t::isnull](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a7878d87e4561a2159a80f4d4ffa157b4)
bool * isnull
**Definition** H5VLconnector.h:832
[H5VL_blob_specific_args_t::op_type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a78d2a3265b57217d686509b4e483d015)
H5VL_blob_specific_t op_type
**Definition** H5VLconnector.h:823
[H5VL_blob_specific_args_t::is_null](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#a9c1b136fbf6bebbfdc1c361d03904a78)
struct H5VL_blob_specific_args_t::@131305020060203175140171235213000033064064140250::@346126041062122026244106230345351003030056115247 is_null
[H5VL_blob_specific_args_t::args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html#ac9e907954c93a08604a9de2f62e1d5f5)
union H5VL_blob_specific_args_t::@131305020060203175140171235213000033064064140250 args
###  blob: optional
Perform a connector-specific operation on a blob via the VOL 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, void *blob_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);  
Arguments:   
obj (IN): Pointer to the blob container. blob_id (IN): Pointer to the blob's connector-specific ID. args (IN/OUT): A pointer to the arguments struct.  
Each connector that requires connector-specific operations should compare the value of the _op_type_ field of the [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) struct with the values returned from calling [H5VLregister_opt_operation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a85d2e5bf7c9e947f5a1645bbd0f887d9) to determine how to handle the optional call and interpret the arguments passed in the struct
##  Token Callbacks
_Structure for token callback routines,[H5VLconnector.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html)_
typedef struct [H5VL_token_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html) {
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[cmp](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a3fd03f11f80b3f6d6beb50585e08e3d5))(void *obj, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token1, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token2, int *cmp_value);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[to_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a16ead8224c9312b1b249ae9e2531d470))(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token, char **token_str);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[from_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a1924e2e9985235f335236fed6a355ebc))(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, const char *token_str, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token);
} [H5VL_token_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html);
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)
**Definition** H5public.h:437
[H5VL_token_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html)
**Definition** H5VLconnector.h:1009
[H5VL_token_class_t::to_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a16ead8224c9312b1b249ae9e2531d470)
herr_t(* to_str)(void *obj, H5I_type_t obj_type, const H5O_token_t *token, char **token_str)
**Definition** H5VLconnector.h:1011
[H5VL_token_class_t::from_str](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a1924e2e9985235f335236fed6a355ebc)
herr_t(* from_str)(void *obj, H5I_type_t obj_type, const char *token_str, H5O_token_t *token)
**Definition** H5VLconnector.h:1012
[H5VL_token_class_t::cmp](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__token__class__t.html#a3fd03f11f80b3f6d6beb50585e08e3d5)
herr_t(* cmp)(void *obj, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value)
**Definition** H5VLconnector.h:1010
###  token: cmp
Compares two tokens and outputs a value like strcmp. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*cmp)(void *obj, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token1, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token2, int *cmp_value);  
Arguments:   
obj (IN): The underlying VOL object. token1 (IN): The first token to compare. token2 (IN): The second token to compare. cmp_value (OUT): A value like strcmp.  
###  token: to_str
Converts a token to a string representation. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*to_str)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token, char **token_str)  
Arguments:   
obj (IN): The underlying VOL object. obj_type (IN): The type of the object. token (IN): The token to turn into a string representation. token_str (OUT): The string representation of the token.  
_The "obj_type" argument is an enum: (from[H5Ipublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html))_
typedef enum [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) {
[H5I_UNINIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bac3bd281c14f862130b9dc14ceb8acbf0) = (-2), 
[H5I_BADID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba3e61c9654de6398dc9676ad37cbe6133) = (-1), 
[H5I_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bacc572b5478629d17dd4fa708c3508f22) = 1, 
[H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6), 
[H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64), 
[H5I_DATASPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bade3bcb0953b041371997f802fa678da8), 
[H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987), 
[H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d), 
[H5I_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5bdc68e9f466027aeac5f8b11205e51f), 
[H5I_VFL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba0d3b691d8e02ae4898c82535401bee05), 
[H5I_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bae950f33b1244e71d24b16786964f04b9), 
[H5I_GENPROP_CLS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9baa5dee573139d32eb67865e1f1405), 
[H5I_GENPROP_LST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bab6161706783d4bca26a889f1ac0cf91a), 
[H5I_ERROR_CLASS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba52573dfc8d6289035fef0757036432d6), 
[H5I_ERROR_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba493c2a05e970214bd4d3aff95fe3f680), 
[H5I_ERROR_STACK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832babfabc2e9a32f38b595f387c5facc7c47), 
[H5I_SPACE_SEL_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5f1a0f50d26adfc30676fc0879cb71ac), 
[H5I_EVENTSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa6512cfec909399be60ac03af2a06724), 
[H5I_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba1008af5d904aebbc78889a8d36bb8836)
} [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b);
The only values which should be used for this call are: 
  * [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6)
  * [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64)
  * [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987)
  * [H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d)


as these are the only objects for which tokens are valid.
###  token: from_str
Converts a string representation of a token to a token. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*from_str)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, const char *token_str, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token);  
Arguments:   
obj (IN): The underlying VOL object. obj_type (IN): The type of the object. token_str (IN): The string representation of the token. token (OUT): The token reated from the string representation.  
_The "obj_type" argument is an enum: (from[H5Ipublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html))_
typedef enum [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) {
[H5I_UNINIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bac3bd281c14f862130b9dc14ceb8acbf0) = (-2), 
[H5I_BADID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba3e61c9654de6398dc9676ad37cbe6133) = (-1), 
[H5I_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bacc572b5478629d17dd4fa708c3508f22) = 1, 
[H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6), 
[H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64), 
[H5I_DATASPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bade3bcb0953b041371997f802fa678da8), 
[H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987), 
[H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d), 
[H5I_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5bdc68e9f466027aeac5f8b11205e51f), 
[H5I_VFL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba0d3b691d8e02ae4898c82535401bee05), 
[H5I_VOL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bae950f33b1244e71d24b16786964f04b9), 
[H5I_GENPROP_CLS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9baa5dee573139d32eb67865e1f1405), 
[H5I_GENPROP_LST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bab6161706783d4bca26a889f1ac0cf91a), 
[H5I_ERROR_CLASS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba52573dfc8d6289035fef0757036432d6), 
[H5I_ERROR_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba493c2a05e970214bd4d3aff95fe3f680), 
[H5I_ERROR_STACK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832babfabc2e9a32f38b595f387c5facc7c47), 
[H5I_SPACE_SEL_ITER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5f1a0f50d26adfc30676fc0879cb71ac), 
[H5I_EVENTSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa6512cfec909399be60ac03af2a06724), 
[H5I_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba1008af5d904aebbc78889a8d36bb8836)
} [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b);
The only values which should be used for this call are: 
  * [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6)
  * [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64)
  * [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987)
  * [H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d)


as these are the only objects for which tokens are valid.
##  Optional Generic Callback
A _generic_ optional callback is provided for services that are specific to a connector. The _optional_ callback has the following definition. It returns an _[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)_ indicating success or failure. 
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*optional)(void *obj, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);  
Arguments:   
obj (IN): A container or object where he operation needs to happen. args (IN/OUT): A pointer to the arguments struct. dxpl_id (IN): The data transfer property list. req IN/OUT): A pointer to the asynchronous request of the operation created by the connector.  
#  New VOL API Routines
API routines have been added to the HDF5 library to manage VOL connectors. This section details each new API call and explains its intended usage. Additionally, a set of API calls that map directly to the VOL callbacks themselves have been added to aid in the development of passthrough connectors which can be stacked and/or split. A list of these API calls is given in an appendix.
##  H5VLpublic.h
The API calls in this header are for VOL management and general use (i.e., not limited to VOL connector authors).
###  H5VLregister_connector_by_name
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1)(const char *connector_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id); [H5VLregister_connector_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf48d1225927e1e701656346b832ee6b1) hid_t H5VLregister_connector_by_name(const char *connector_name, hid_t vipl_id) Registers a new VOL connector by name.  
Arguments:   
connector_name (IN): The connector name to search for and register. vipl_id (IN): An ID for a VOL initialization property list (vipl).  
Registers a VOL connector with the HDF5 library given the name of the connector and returns an identifier for it ([H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on errors). If the connector is already registered, the library will create a new identifier for it and returns it to the user; otherwise the library will search the plugin path for a connector of that name, loading and registering it, returning an ID for it, if found. See the [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html) for more information on loading plugins and the search paths.
###  H5VLregister_connector_by_value
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8)([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id); [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) int H5VL_class_value_t VOL connector identifiers. **Definition** H5VLpublic.h:175 [H5VLregister_connector_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga11e69930e47f654805a265f417412ea8) hid_t H5VLregister_connector_by_value(H5VL_class_value_t connector_value, hid_t vipl_id) Registers a new VOL connector by value.  
Arguments:   
connector_value (IN): The connector value to search for and register. vipl_id (IN): An ID for a VOL initialization property list (vipl).  
Registers a VOL connector with the HDF5 library given a value for the connector and returns an identifier for it ([H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on errors). If the connector is already registered, the library will create a new identifier for it and returns it to the user; otherwise the library will search the plugin path for a connector of that name, loading and registering it, returning an ID for it, if found. See the VOL User Guide for more information on loading plugins and the search paths.
###  H5VLis_connector_registered_by_name
Signature:   
---  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5VLis_connector_registered_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca)(const char *name); [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) int htri_t **Definition** H5public.h:282 [H5VLis_connector_registered_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9be3c92e4430b9cf42a376534a47fcca) htri_t H5VLis_connector_registered_by_name(const char *name) Tests whether a VOL class has been registered under a certain name.  
Arguments:   
name (IN): The connector name to check for.  
Checks if a VOL connector is registered with the library given the connector name and returns TRUE/FALSE on success, otherwise it returns a negative value.
###  H5VLis_connector_registered_by_value
Signature:   
---  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5VLis_connector_registered_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b)([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value); [H5VLis_connector_registered_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga83ba8986ed68f67c41b492dfd273804b) htri_t H5VLis_connector_registered_by_value(H5VL_class_value_t connector_value) Tests whether a VOL class has been registered for a given value.  
Arguments:   
connector_value (IN): The connector value to check for.  
Checks if a VOL connector is registered with the library given the connector value and returns TRUE/FALSE on success, otherwise it returns a negative value.
###  H5VLget_connector_id
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLget_connector_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga5b69c29931e55208517c598ac3039f77)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id); [H5VLget_connector_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga5b69c29931e55208517c598ac3039f77) hid_t H5VLget_connector_id(hid_t obj_id) Retrieves the VOL connector identifier for a given object identifier.  
Arguments:   
obj_id (IN): An ID for an HDF5 VOL object.  
Given a VOL object such as a dataset or an attribute, this function returns an identifier for its associated connector. If the ID is not a VOL object (such as a dataspace or uncommitted datatype), [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) is returned. The identifier must be released with a call to [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.").
###  H5VLget_connector_id_by_name
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLget_connector_id_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gabcbf9b9b07a6b60e17ff9681684f944d)(const char *name); [H5VLget_connector_id_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gabcbf9b9b07a6b60e17ff9681684f944d) hid_t H5VLget_connector_id_by_name(const char *name) Retrieves the identifier for a registered VOL connector name.  
Arguments:   
name (IN): The connector name to check for.  
Given a connector name that is registered with the library, this function returns an identifier for the connector. If the connector is not registered with the library, [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) is returned.The identifier must be released with a call to [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.").
###  H5VLget_connector_id_by_value
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLget_connector_id_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee)([H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) connector_value); [H5VLget_connector_id_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga8f6d366bc6b8323bbffe1e5a5ba18bee) hid_t H5VLget_connector_id_by_value(H5VL_class_value_t connector_value) Retrieves the identifier for a registered VOL connector value.  
Arguments:   
connector_value (IN): The connector value to check for.  
Given a connector value that is registered with the library, this function returns an identifier for the connector. If the connector is not registered with the library, [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) is returned.The identifier must be released with a call to [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier.").
###  H5VLget_connector_name
Signature:   
---  
ssize_t [H5VLget_connector_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, char *name /*out*/, size_t size); [H5VLget_connector_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaf326406d7733c0ab8d12118c13c78dfa) ssize_t H5VLget_connector_name(hid_t id, char *name, size_t size) Retrieves a connector name for a VOL.  
Arguments:   
id (IN): The object identifier to check. name (OUT): Buffer pointer to put the connector name. If NULL, the library just returns thesize required to store the connector name. size (IN): the size of the passed in buffer.  
Retrieves the name of a VOL connector given an object identifier that was created/opened ith it. On success, the name length is returned.
###  H5VLclose
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id); [H5VLclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094) herr_t H5VLclose(hid_t connector_id) Closes a VOL connector identifier.  
Arguments:   
connector_id (IN): A valid identifier of the connector to close.  
Shuts down access to the connector that the identifier points to and release resources associated with it.
###  H5VLunregister_connector
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLunregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id); [H5VLunregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaffbdc22f724c2c818f3be3845145d73e) herr_t H5VLunregister_connector(hid_t connector_id) Removes a VOL connector identifier from the library.  
Arguments:   
connector_id (IN): A valid identifier of the connector to unregister.  
Unregisters a connector from the library and return a positive value on success otherwise return a negative value. The native VOL connector cannot be unregistered (this will return a negative [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) value).
###  H5VLquery_optional
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLquery_optional](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, int opt_type, uint64_t *flags); [H5VLquery_optional](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga17ef00e528d99eda5879d749c2a12043) herr_t H5VLquery_optional(hid_t obj_id, H5VL_subclass_t subcls, int opt_type, uint64_t *flags) Determine if a VOL connector supports a particular optional callback operation.  
Arguments:   
obj_id (IN): A valid identifier of a VOL-managed object. subcls (IN): The subclass of the optional operation. opt_type (IN): The optional operation. The native VOL connector uses hard-coded values. Other VOL connectors get this value when the optional operations are registered. flags (OUT): Bitwise flags indicating support and behavior.  
Determines if a connector or connector stack (determined from the passed-in object) supports an optional operation. The returned flags (listed below) not only indicate whether the operation is supported or not, but also give a sense of the option's behavior (useful for pass-through connectors).
Bitwise query flag values: 
#define H5VL_OPT_QUERY_SUPPORTED 0x0001 /* VOL connector supports this operation */
#define H5VL_OPT_QUERY_READ_DATA 0x0002 /* Operation reads data for object */
#define H5VL_OPT_QUERY_WRITE_DATA 0x0004 /* Operation writes data for object */
#define H5VL_OPT_QUERY_QUERY_METADATA 0x0008 /* Operation reads metadata for object */
#define H5VL_OPT_QUERY_MODIFY_METADATA 0x0010 /* Operation modifies metadata for object */
#define H5VL_OPT_QUERY_COLLECTIVE 0x0020 /* Operation is collective (operations without this flag are assumed to be independent) */
#define H5VL_OPT_QUERY_NO_ASYNC 0x0040 /* Operation may NOT be executed asynchronously */
#define H5VL_OPT_QUERY_MULTI_OBJ 0x0080 /* Operation involves multiple objects */
##  H5VLconnector.h
This functionality is intended for VOL connector authors and includes helper functions that are useful for writing connectors.
API calls to manage optional operations are also found in this header file. These are discussed in the section on optional operations, above.
###  H5VLregister_connector
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b)(const [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) *cls, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id); [H5VLregister_connector](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga439c150299522a0e0f401a86d083097b) hid_t H5VLregister_connector(const H5VL_class_t *cls, hid_t vipl_id) Registers a new VOL connector.  
Arguments:   
cls (IN): A pointer to the connector structure to register. vipl_id (IN): An ID for a VOL initialization property list (vipl).  
Registers a user-defined VOL connector with the HDF5 library and returns an identifier for that connector ([H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on errors). This function is used when the application has direct access to the connector it wants to use and is able to obtain a pointer for the connector structure to pass to the HDF5 library.
###  H5VLobject
Signature:   
---  
void *[H5VLobject](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga21f351a8a3a128659f57217a3b452cd5)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id); [H5VLobject](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga21f351a8a3a128659f57217a3b452cd5) void * H5VLobject(hid_t obj_id)  
Arguments:   
obj_id (IN): identifier of the object to dereference.  
Retrieves a pointer to the VOL object from an HDF5 file or object identifier.
###  H5VLget_file_type
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLget_file_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga161553978d3d001a5b04708acccb429f)(void *file_obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id); [H5VLget_file_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#ga161553978d3d001a5b04708acccb429f) hid_t H5VLget_file_type(void *file_obj, hid_t connector_id, hid_t dtype_id)  
Arguments:   
file_obj (IN): pointer to a file or file object's connector-specific data. connector_id (IN): A valid identifier of the connector to use. dtype_id (IN): A valid identifier for the type.  
Returns a copy of the _dtype_id_ parameter but with the location set to be in the file. Returns a negative value ([H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) on errors.
##  H5VLconnector_passthru.h
This functionality is intended for VOL connector authors who are writing pass-through connectors and includes helper functions that are useful for writing such connectors. Callback equivalent functions can be found in this header as well. A list of these functions is included as an appendix to this document.
###  H5VLcmp_connector_cls
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLcmp_connector_cls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a814ba230b852c6b7e27811fe2b8d4fb2)(int *cmp, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id1, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id2); [H5VLcmp_connector_cls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a814ba230b852c6b7e27811fe2b8d4fb2) herr_t H5VLcmp_connector_cls(int *cmp, hid_t connector_id1, hid_t connector_id2)  
Arguments:   
cmp (OUT): a value like strcmp. connector_id1 (IN): the ID of the first connector to compare. connector_id2 (IN): the ID of the second connector to compare  
Compares two connectors (given by the connector IDs) to see if they refer to the same connector underneath. Returns a positive value on success and a negative value on errors.
###  H5VLwrap_register
Signature:   
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5VLwrap_register](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9873d50b395911b609621c22c2fa554b)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type); [H5VLwrap_register](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#ga9873d50b395911b609621c22c2fa554b) hid_t H5VLwrap_register(void *obj, H5I_type_t type) Wrap an internal object with a "wrap context" and register an hid_t for the resulting object.  
Arguments:   
obj (IN): an object to wrap. type (IN): the type of the object (see below).  
Wrap an internal object with a "wrap context" and register and return an hidt for the resulting object. This routine is mainly targeted toward wrapping objects for iteration routine callbacks (i.e. the callbacks from H5Aiterate*, H5Literate* / H5Lvisit*, and H5Ovisit* ). Using it in an application will return an error indicating the API context isn't available or can't be retrieved. he type must be a VOL-managed object class: 
  * [H5I_FILE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bacc572b5478629d17dd4fa708c3508f22)
  * [H5I_GROUP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa839c547a95f216c36697065422162d6)
  * [H5I_DATATYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf881cdc68cc4082e66091f0b4bfb9e64)
  * [H5I_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baa9f2e1d8a2db4f302d81603217b83987)
  * [H5I_MAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832baf61d30fecc42d847825922bc97de1b0d)
  * [H5I_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba5bdc68e9f466027aeac5f8b11205e51f)


###  H5VLretrieve_lib_state
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLretrieve_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4bb2aa208a7d36a1da7f51639f73c22f)(void **state); [H5VLretrieve_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4bb2aa208a7d36a1da7f51639f73c22f) herr_t H5VLretrieve_lib_state(void **state)  
Arguments:   
state (OUT): the library state.  
Retrieves a copy of the internal state of the HDF5 library, so that it can be restored later. Returns a positive value on success and a negative value on errors.
###  H5VLopen_lib_context
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLopen_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d)(void **context); [H5VLopen_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d) herr_t H5VLopen_lib_context(void **context) Opens a new internal context for the HDF5 library.  
Opens a new internal context for the HDF5 library. Returns a non-negative value if `*context` is set; otherwise returns negative value if `*context` is unset.
###  H5VLrestore_lib_state
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrestore_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a160ba58761792ce81e6951e517940fdc)(const void *state); [H5VLrestore_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a160ba58761792ce81e6951e517940fdc) herr_t H5VLrestore_lib_state(const void *state)  
Arguments:   
state (IN): the library state.  
Restores the internal state of the HDF5 library. Returns a positive value on success and a negative value on errors.
###  H5VLclose_lib_context
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLclose_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae05f169607f8be714fd34424c3786ee0)(void *context); [H5VLclose_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae05f169607f8be714fd34424c3786ee0) herr_t H5VLclose_lib_context(void *context) Closes the internal state of the HDF5 library.  
Closes the state of the library, undoing the effects of [H5VLopen_lib_context](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_v.html#gae3b0dc77bc58f9d84cbeb733da1cb94d "Opens a new internal context for the HDF5 library."). Returns a positive value on success and a negative value on errors.
###  H5VLfree_lib_state
Signature:   
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfree_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a37856e5f0914916c4ac36cbc0e6bed84)(void *state); [H5VLfree_lib_state](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a37856e5f0914916c4ac36cbc0e6bed84) herr_t H5VLfree_lib_state(void *state)  
Arguments:   
state (IN): the library state.  
Free a retrieved library state. Returns a positive value on success and a negative value on errors.
#  Appendix A Mapping of VOL Callbacks to HDF5 API Calls
VOL Callback  | HDF5 API Call   
---|---  
FILE   
create  | 
  * [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.")

  
open  | 
  * [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.")

  
get  | 
  * [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4 "Returns a file access property list identifier.")
  * [H5Fget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga2f823a9e929b00b06a6be80619a61778 "Returns a file creation property list identifier.")
  * [H5Fget_fileno](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga402205688af065ab5db0fe20417d5484 "Retrieves a file's file number that uniquely identifies an open file.")
  * [H5Fget_intent](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga466179d7783d256329c2e3110055a16c "Determines the read/write or read-only status of a file.")
  * [H5Fget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga0ed43dbe476a160b73f55127c7db797c "Retrieves name of file to which object belongs.")
  * [H5Fget_obj_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadcdae0aca7c88064db0d32de7f1e31f2 "Returns the number of open object identifiers for an open file.")
  * [H5Fget_obj_ids](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga35e72579bd07433162b80ddc0bd0c5b1 "Returns a list of open object identifiers.")

  
specific  | 
  * [H5Fdelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga2e8b5e19b343123e8ab21442f9169a62 "Deletes an HDF5 file.")
  * [H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.")
  * [H5Fis_accessible](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga584471c3b98453b9b04a4bf9af847442 "Checks if a file can be opened with a given file access property list.")
  * [H5Fis_hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225 "Determines whether a file is in the HDF5 format.") (deprecated, hard-coded to use native connector) 
  * [H5Freopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3f213eb05c5419d63ba168c30036e47b "Returns a new identifier for a previously-opened HDF5 file.")

  
close  | 
  * [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.")

  
GROUP   
create  | 
  * [H5Gcreate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga7397440085510728a2e2d22199e81980 "Creates a new group and links it into the file.") (deprecated) 
  * [H5Gcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga86d93295965f750ef25dea2505a711d9 "Creates a new group and links it into the file.")
  * [H5Gcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gab52641f0736281faaaae4e3039bbb344 "Creates a new empty group without linking it into the file structure.")

  
open  | 
  * [H5Gopen1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga163ca3eb7893d34973ee900b2da886be "Opens an existing group for modification and returns a group identifier for that group.") (deprecated) 
  * [H5Gopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadab91e2dd7a7e253dcc0e4fe04b81403 "Opens an existing group in a file.")

  
get  | 
  * [H5Gget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga0b959a53cbffa48f5d68ce33b43b7ed8 "Gets a group creation property list identifier.")
  * [H5Gget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad4be126ab7bbf2001435e8e70089f3d3 "Retrieves information about a group.")
  * [H5Gget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga985f27ad1a164d99fa1f58c6de60ab00 "Retrieves information about a group, according to the group's position within an index.")
  * [H5Gget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadedd0c73c98f2ada69305f2992c3300e "Retrieves information about a group by its name.")
  * [H5Gget_num_objs](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3e30142e15ccf9a08bfc91ca9925c14d "Returns number of objects in the group specified by its identifier.") (deprecated) 

  
specific  | 
  * [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.")
  * [H5Funmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga79e80cbf6d9782402b353b763162779c "Un-mounts an HDF5 file.")
  * [H5Gflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga1d55dbf931f8003bb329c4340b8fe4d6 "Flushes all buffers associated with a group to disk.")
  * [H5Grefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga0a8bdd0eb1b001222c27d3d39a909840 "Refreshes all buffers associated with a group.")

  
close  | 
  * [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.")

  
DATASET   
create  | 
  * [H5Dcreate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.") (deprecated) 
  * [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file.")

  
open  | 
  * [H5Dopen1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81 "Opens an existing dataset.") (deprecated) 
  * [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset.")

  
read  | 
  * [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.")

  
write  | 
  * [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.")

  
get  | 
  * [H5Dget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b "Returns the dataset access property list associated with a dataset.")
  * [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163 "Returns an identifier for a copy of the dataset creation property list for a dataset.")
  * [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.")
  * [H5Dget_space_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.")
  * [H5Dget_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.")
  * [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.")

  
specific  | 
  * [H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91 "Extends a dataset.") (deprecated) 
  * [H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7 "Flushes all buffers associated with a dataset to disk.")
  * [H5Drefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da "Refreshes all buffers associated with a dataset.")
  * [H5Dset_extent](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.")

  
close  | 
  * [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.")

  
OBJECT   
open  | 
  * [H5Oopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9f635f58c7ddf17f87c253bfbca08bc1 "Opens an object in an HDF5 file by location identifier and path name.")
  * [H5Oopen_by_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga137f3823adab4daaaf8fe87b40453fa2 "Opens an object using its address within an HDF5 file.") (deprecated) 
  * [H5Oopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb66e5cbb3ca79890fc284a0b06762be "Opens the nth object in a group.")
  * [H5Oopen_by_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2ea3627cf171d0565307702a5e203262 "Opens an object in an HDF5 file using its VOL independent token.")

  
copy  | 
  * [H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")

  
get  | 
  * [H5Oget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80 "Retrieves the metadata for an object specified by an identifier.") (deprecated) 
  * [H5Oget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga06f896e14fe4fa940fbc2bc235e0cf74 "Retrieves the metadata for an object specified by an identifier.") (deprecated) 
  * [H5Oget_info3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013 "Retrieves the metadata for an object specified by an identifier.")

  
specific  | 
  * [H5Odecr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga60c20da5e244c28a653d4fa23d316b44 "Decrements an object reference count.")
  * [H5Oexists_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gab0fef18d97844c4f83d412c5a22def7b "Determines whether a link resolves to an actual object.")
  * [H5Oflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gad99f35048cba4534b6393214684f090f "Flushes all buffers associated with an HDF5 object to disk.")
  * [H5Oincr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2086bad6c3cd2a711c306a48c093ff55 "Increments an object reference count.")
  * [H5Orefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0318b68be9ab23a92b8a6bee0af9e2f "Refreshes all buffers associated with an HDF5 object.")
  * [H5Ovisit_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaffacf3bd66f4fe074099eae1c80914f2 "Recursively visits all objects starting from a specified object.") (deprecated) 
  * [H5Ovisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9c155caf5499405fe403e1eb27b5beb6 "Recursively visits all objects starting from a specified object.") (deprecated) 
  * [H5Ovisit_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga34815400b01df59c4dac19436124885a "Recursively visits all objects accessible from a specified object.")
  * [H5Ovisit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6efdb2a0a9fe9fe46695cc0f7bd993e7 "Recursively visits all objects accessible from a specified object.") (deprecated) 
  * [H5Ovisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa4ab542f581f4fc9a4eaa95debb29c9e "Recursively visits all objects accessible from a specified object.") (deprecated) 
  * [H5Ovisit3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a "Recursively visits all objects accessible from a specified object.")

  
close  | 
  * [H5Oclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga545ad7c54987013ebd50b40fe9e73c61 "Closes an object in an HDF5 file.")

  
LINK   
create  | 
  * [H5Glink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga1b9b2effdc1727613f81c4dcb2a4d644 "Creates a link of the specified type from new_name to cur_name.") (deprecated) 
  * [H5Glink2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gafabd07a7f64a7cbef27c56a3bee2df47 "Creates a link of the specified type from cur_name to new_name.") (deprecated) 
  * [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object.")
  * [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.")
  * [H5Lcreate_ud](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2 "Creates a link of a user-defined type.")
  * [H5Olink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2c97dd58e64b67d16325fceb7e02113f "Creates a hard link to an object in an HDF5 file.")

  
copy  | 
  * [H5Lcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...")

  
move  | 
  * [H5Gmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gaa6474351d346ad45309ae0b22ebdde9a "Renames an object within an HDF5 file.") (deprecated) 
  * [H5Gmove2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad97bf21798b06b63df0bdd404cac562c "Renames an object within an HDF5 file.") (deprecated) 
  * [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.")

  
get  | 
  * [H5Gget_linkval](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3a994ec16caa60edd7bb6c71c6fdc5aa "Returns the name of the object that the symbolic link points to.") (deprecated) 
  * [H5Lget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.") (deprecated) 
  * [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.")
  * [H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67)
  * [H5Lget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.")
  * [H5Lget_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.")
  * [H5Lget_val_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c "Retrieves value of the n-th link in a group, according to the order within an index.")

  
specific  | 
  * [H5Gunlink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gacb843cbd5bbb816cfa9c855463d1e51c "Removes the link to an object from a group.") (deprecated) 
  * [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.")
  * [H5Ldelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805 "Removes the n-th link in a group.")
  * [H5Lexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.")
  * [H5Literate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1 "Iterates over links in a group, with user callback routine, according to the order within an index.") (deprecated) 
  * [H5Literate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index.")
  * [H5Literate_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga87e036da0c8d1146a073f3ee08e0fedc "Iterates through links in a group by its name.") (deprecated) 
  * [H5Literate_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e "Iterates through links in a group.")
  * [H5Lvisit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga5424ef7043c82147490d027a0e8a59ef "Recursively visits all links starting from a specified group.") (deprecated) 
  * [H5Lvisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66 "Recursively visits all links starting from a specified group.")
  * [H5Lvisit_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1f1ba1bb4d44f2c111990024809417ac "Recursively visits all links starting from a specified group.") (deprecated) 
  * [H5Lvisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173 "Recursively visits all links starting from a specified group.")

  
DATATYPE   
commit  | 
  * [H5Tcommit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1c00afb6dc5534778370a92c33fa2625 "Commits a transient datatype to a file, creating a newly named datatype.") (deprecated) 
  * [H5Tcommit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga10352b6fa9ac58a7fbd5299496f1df31 "Commits a transient datatype, linking it into the file and creating a new committed datatype.")
  * [H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd)+anon 

  
open  | 
  * [H5Topen1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9f76fa0dc34bc7b310e100e5bfed66fb "Opens a named datatype.") (deprecated) 
  * [H5Topen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga7e65e77634f1fb4ba38cbcdab9a59bc2 "Opens a committed \(named\) datatype.")

  
get  | 
  * [H5Tget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga6802c22c6e90216aa839a4a83909a54c "Returns a copy of a datatype's creation property list.")

  
specific  | 
  * [H5Tflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafd60389b49e1e5e6f37caffbe6cbf6e5 "Flushes all buffers associated with a committed datatype to disk.")
  * [H5Trefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga5bc56f6b85e114829dc12d6b18d66f4d "Refreshes all buffers associated with a committed datatype.")

  
close  | 
  * [H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0 "Releases a datatype.")

  
ATTRIBUTE   
create  | 
  * [H5Acreate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa30f5f6c277d6c46f8aa31e89cdba085 "Creates an attribute attached to a specified object.") (deprecated) 
  * [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308 "Creates an attribute attached to a specified object.")
  * [H5Acreate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga004160c28e281455ec48aa7fe557ef8a "Creates an attribute attached to a specified object.")

  
open  | 
  * [H5Aopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga59863b205b6d93b2145f0fbca49656f7 "Opens an attribute for an object specified by object identifier and attribute name.")
  * [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d "Opens the nth attribute attached to an object.")
  * [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10 "Opens an attribute for an object by object name and attribute name.")
  * [H5Aopen_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadaa85276f2731ad78462a6fd27118470 "Opens the attribute specified by its index.") (deprecated) 
  * [H5Aopen_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga5c05fade96b6b7e2299f56a5b1edb1c1 "Opens an attribute specified by name.") (deprecated) 

  
read  | 
  * [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c "Reads the value of an attribute.")

  
write  | 
  * [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c "Writes data to an attribute.")

  
get  | 
  * [H5Aget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0f6b545850bd21f128904eff51df226d "Gets an attribute creation property list identifier.")
  * [H5Aget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gae3f1b7b87240b461f7827a8783acc08a "Retrieves attribute information by attribute identifier.")
  * [H5Aget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gad110910cb227c15fdca938a642714fe9 "Retrieves attribute information by attribute index position.")
  * [H5Aget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga258f03e12b4f49ad33ba72d17a9e2faf "Retrieves attribute information by attribute name.")
  * [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674 "Gets an attribute name.")
  * [H5Aget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4c552b2db32371f8ea20d87475313fb6 "Gets an attribute name by attribute index position.")
  * [H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d "Gets a copy of the dataspace for an attribute.")
  * [H5Aget_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabd11c8e11db0adde706e41a24a832f06 "Returns the amount of storage used to store an attribute.")
  * [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44 "Gets an attribute's datatype.")

  
specific  | 
  * [H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba "Deletes an attribute from a specified location.")
  * [H5Adelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga06711a4e77ff8ab49e427010fd38ac9e "Deletes an attribute from an object according to index order.")
  * [H5Adelete_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gacbf689308f851428dd641b64f5f94feb "Removes an attribute from a specified location.")
  * [H5Aexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga293b5be270d90cd5e47f782ca9aec80b "Determines whether an attribute with a given name exists on an object.")
  * [H5Aexists_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa1d2305651a4524f6aa0f8b56eec1a37 "Determines whether an attribute with a given name exists on an object.")
  * [H5Aiterate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.") (deprecated) 
  * [H5Aiterate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251 "Calls a user-defined function for each attribute on an object.")
  * [H5Aiterate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga75db973d69b61f673f5cdf21ac624cef "Calls user-defined function for each attribute on an object.")
  * [H5Arename](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga490dcd6db246c1fda7295badfce28203 "Renames an attribute.")
  * [H5Arename_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga21f8483c935d72187b98f5e7c2056140)

  
close  | 
  * [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda "Closes the specified attribute.")

  
#  Appendix B Callback Wrapper API Calls for Passthrough Connector Authors
/* Pass-through callbacks */
H5_DLL void *[H5VLget_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a542800ebb029002c3f82f4efe2d50ee3)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLget_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad4d06542aef57546ccd1e40c4955e03e)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void **wrap_ctx);
H5_DLL void *[H5VLwrap_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aac5db40d46b03bf31d9cee91fdb5ca14)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void *wrap_ctx);
H5_DLL void *[H5VLunwrap_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a613986d72333d30e5487ae334c8df0a1)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfree_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#af769237a892ee3c20d7cd75d22a64fc6)(void *wrap_ctx, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id);
/* Public wrappers for generic callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLinitialize](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a8e43a9640d599a68b2ce281796920d88)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) vipl_id);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLterminate](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a45e9fa8c9a6c037aed0b0521bb884148)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLget_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4ad793093a03375e7af24f27bc60c2e5)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, uint64_t *cap_flags);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLget_value](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#adc2fe50b5e8945e1ba778c05118381f2)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga81b40d59b53c498f8aa9d92d0afdde2c) *conn_value);
/* Public wrappers for info fields and callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLcopy_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a910252aef3ceccad24ccc1cd03a38450)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void **dst_vol_info, void *src_vol_info);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLcmp_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a19b87561296be6fb895fd123df3dc972)(int *cmp, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const void *info1, const void *info2);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfree_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d7c204a3db83d5563b0be557a3a4571)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void *vol_info);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLconnector_info_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aac2b19b03066f3f9e07aae264de6bd14)(const void *info, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, char **str);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLconnector_str_to_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f)(const char *str, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void **info);
/* Public wrappers for attribute callbacks */
H5_DLL void *[H5VLattr_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae52e1eebbdd8dc7ceb179ab694552a98)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL void *[H5VLattr_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae795c2b22a3c8c610770332082d5d567)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_read](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad5a4f0da0e12b4f37df203c982687bdb)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_write](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac40f6f460fe5bc4de1db2f5ee7bdc647)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0455ecaf77b7e60008bdc5aedb748e66)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_attr_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac632a877cce1103f4e599959d4cc5460)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_attr_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__attr__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a29af736e7016e0d218f14d5e706794f5)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLattr_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6f5ab5c95feb563669c61e71a1af79c9)(void *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for dataset callbacks */
H5_DLL void *[H5VLdataset_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ab2b71f560cd6112c4b07de94335b30e8)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL void *[H5VLdataset_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a43cee299f5913ddee33cd3a855da9048)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_read](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9e8670dd258b866391da0633a823a01d)(size_t count, void *dset[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[],
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, void *buf[],
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_write](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a5028517e60ff4ae4a34a6c9ff1185668)(size_t count, void *dset[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[],
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id,
const void *buf[], void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a7cd52e5b61d504e7d6e3a769534efdc7)(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_dataset_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aff9b695f422d86e9aff84a165acd2658)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_dataset_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__dataset__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9311dc565e5286c9f6e7d6594cf55781)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdataset_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a678c02d9005a68920ca71c8078748fb5)(void *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for named datatype callbacks */
H5_DLL void *[H5VLdatatype_commit](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac3f34096fc3f8a0e939eb479bdbd1f38)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL void *[H5VLdatatype_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a83e32eb893ad112df89d304bb32c3cea)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) tapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdatatype_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#adafa64999e4dda2540843fe333a6a884)(void *dt, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_datatype_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdatatype_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6d9af2589e98fb0bc536a6dd1bc36ab1)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_datatype_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__datatype__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdatatype_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ab0de23d096ca426c2c7c81c22ca6ccf7)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLdatatype_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a49f02bbb1985181993a530f109b33707)(void *dt, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for file callbacks */
H5_DLL void *[H5VLfile_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#abedac242a6b05c74630aee6717c6eb86)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL void *[H5VLfile_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a5c60fb975ab92946e798c29d4490bc33)(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfile_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3c548e9cb2bd51331ee897aa86f21a46)(void *file, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_file_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfile_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3eeb23a2c3687d694a6dd0106bf34820)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_file_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__file__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfile_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae9801ec2575976a0b5ed461ecb0b3689)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLfile_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a2c0def6910a6f7c52de27e0bb8d7a35c)(void *file, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for group callbacks */
H5_DLL void *[H5VLgroup_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6e872a3063e66fae44331ad427c11f28)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL void *[H5VLgroup_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#acd34999e825ada4964f0d4d367cad42f)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLgroup_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a8f16a678c422196f5b269fcf64eb0f57)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_group_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLgroup_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aaa255508c8fd64a8f24ab66497ac3800)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_group_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__group__specific__args__t.html) *args,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLgroup_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a2735899439c85a23e1a452ae980b9782)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLgroup_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#afa20af338f0722587bec9af00e7a041c)(void *grp, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for link callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0f65e11e4b66231e352093c97bbc8f3c)([H5VL_link_create_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__create__args__t.html) *args, void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3d24ef9796e673c78a7220edff4919ae)(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj,
const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_move](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a09c1f3374d3b0aeabec554a8d63a082a)(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, void *dst_obj,
const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac87aec9385917d6f0d572de479f0bdb6)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_link_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4d8d57fd847725b7f25817d2613858d7)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_link_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__link__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLlink_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aa493c236d15cd60a610fbb4821e8d095)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for object callbacks */
H5_DLL void *[H5VLobject_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3334754bfe6ae2b1d0eaaa9ee2aca7ed)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) *opened_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLobject_copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad3e5984fa0bebddfe9d38e8ba0257ceb)(void *src_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params1, const char *src_name,
void *dst_obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params2, const char *dst_name,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLobject_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a1d9c6243482bc42471e3be9393fb61d1)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_object_get_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__get__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLobject_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a91f97ca5ec160ed4f2bff2fc1949cddd)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_object_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__object__specific__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLobject_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a07bdc0a6e611e80240d610e25a4dc165)(void *obj, const [H5VL_loc_params_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__loc__params__t.html) *loc_params, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id,
[H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void **req);
/* Public wrappers for connector/container introspection callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLintrospect_get_conn_cls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a30c55c91df126248b0bf83e06a4c4cb8)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_get_conn_lvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a72dd04b7264916fe5cdfc5970fe8ae21) lvl,
const [H5VL_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__class__t.html) **conn_cls);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLintrospect_get_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a202087e28ac183af87331010965b9616)(const void *info, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, uint64_t *cap_flags);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLintrospect_opt_query](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3e1f197a5a90c4f3c5a8fb6ff6031cd6)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_subclass_t](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_d_e_f.html#ga14175ca7d867657e3e5c2f79a154a599) subcls, int opt_type,
uint64_t *flags);
/* Public wrappers for asynchronous request callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_wait](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a7c04eeb53490737229ec93252b86e2f2)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, uint64_t timeout,
[H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_notify](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9b490e7864366df596270c003af74468)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_request_notify_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a418e1ef08bd10c57dc12f04b1e22f784) cb, void *ctx);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_cancel](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae125d77504bfc8ba5edf68a5d1796856)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_request_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector_8h.html#a9861877746c10d523dc8d5148f18ac3b) *status);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a56983158d53c6f71ee5c7613f96265ef)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_request_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__request__specific__args__t.html) *args);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d4fc41090ed919e7f134dcdd8bc2356)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLrequest_free](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4bdb448184824a25bb3f7ccecd99444b)(void *req, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id);
/* Public wrappers for blob callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLblob_put](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a069eba2bccdc85798a789ce2bd0faeb6)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const void *buf, size_t size, void *blob_id,
void *ctx);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLblob_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aecf782e83423e79fa4660c0d1b1978a8)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const void *blob_id, void *buf, size_t size,
void *ctx);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLblob_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4824cecde7ca9cdcee9df2f0db3c288a)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void *blob_id,
[H5VL_blob_specific_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__blob__specific__args__t.html) *args);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLblob_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a24bda37b03b8caf0406d3822c23465df)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, void *blob_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args);
/* Public wrappers for token callbacks */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLtoken_cmp](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a91cd262c9fced15807636937f8ae91b6)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token1,
const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token2, int *cmp_value);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLtoken_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a497f2cf9aefaba6f335be5a32dc3e109)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token,
char **token_str);
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLtoken_from_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a1bdbf39a88dd00bb47a32eb1264df39b)(void *obj, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) obj_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, const char *token_str,
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token);
/* Public wrappers for generic 'optional' callback */
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5VLoptional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aeeacac4f0290962703435fd2f4b794be)(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) connector_id, [H5VL_optional_args_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_v_l__optional__args__t.html) *args, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id,
void **req);
[H5VLattr_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0455ecaf77b7e60008bdc5aedb748e66)
herr_t H5VLattr_get(void *obj, hid_t connector_id, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLblob_put](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a069eba2bccdc85798a789ce2bd0faeb6)
herr_t H5VLblob_put(void *obj, hid_t connector_id, const void *buf, size_t size, void *blob_id, void *ctx)
[H5VLobject_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a07bdc0a6e611e80240d610e25a4dc165)
herr_t H5VLobject_optional(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLlink_move](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a09c1f3374d3b0aeabec554a8d63a082a)
herr_t H5VLlink_move(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t connector_id, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
[H5VLrequest_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d4fc41090ed919e7f134dcdd8bc2356)
herr_t H5VLrequest_optional(void *req, hid_t connector_id, H5VL_optional_args_t *args)
[H5VLfree_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0d7c204a3db83d5563b0be557a3a4571)
herr_t H5VLfree_connector_info(hid_t connector_id, void *vol_info)
[H5VLlink_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a0f65e11e4b66231e352093c97bbc8f3c)
herr_t H5VLlink_create(H5VL_link_create_args_t *args, void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
[H5VLcmp_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a19b87561296be6fb895fd123df3dc972)
herr_t H5VLcmp_connector_info(int *cmp, hid_t connector_id, const void *info1, const void *info2)
[H5VLtoken_from_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a1bdbf39a88dd00bb47a32eb1264df39b)
herr_t H5VLtoken_from_str(void *obj, H5I_type_t obj_type, hid_t connector_id, const char *token_str, H5O_token_t *token)
[H5VLobject_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a1d9c6243482bc42471e3be9393fb61d1)
herr_t H5VLobject_get(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLintrospect_get_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a202087e28ac183af87331010965b9616)
herr_t H5VLintrospect_get_cap_flags(const void *info, hid_t connector_id, uint64_t *cap_flags)
[H5VLblob_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a24bda37b03b8caf0406d3822c23465df)
herr_t H5VLblob_optional(void *obj, hid_t connector_id, void *blob_id, H5VL_optional_args_t *args)
[H5VLgroup_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a2735899439c85a23e1a452ae980b9782)
herr_t H5VLgroup_optional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLattr_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a29af736e7016e0d218f14d5e706794f5)
herr_t H5VLattr_optional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLfile_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a2c0def6910a6f7c52de27e0bb8d7a35c)
herr_t H5VLfile_close(void *file, hid_t connector_id, hid_t dxpl_id, void **req)
[H5VLintrospect_get_conn_cls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a30c55c91df126248b0bf83e06a4c4cb8)
herr_t H5VLintrospect_get_conn_cls(void *obj, hid_t connector_id, H5VL_get_conn_lvl_t lvl, const H5VL_class_t **conn_cls)
[H5VLobject_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3334754bfe6ae2b1d0eaaa9ee2aca7ed)
void * H5VLobject_open(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5I_type_t *opened_type, hid_t dxpl_id, void **req)
[H5VLfile_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3c548e9cb2bd51331ee897aa86f21a46)
herr_t H5VLfile_get(void *file, hid_t connector_id, H5VL_file_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLlink_copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3d24ef9796e673c78a7220edff4919ae)
herr_t H5VLlink_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t connector_id, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
[H5VLintrospect_opt_query](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3e1f197a5a90c4f3c5a8fb6ff6031cd6)
herr_t H5VLintrospect_opt_query(void *obj, hid_t connector_id, H5VL_subclass_t subcls, int opt_type, uint64_t *flags)
[H5VLfile_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a3eeb23a2c3687d694a6dd0106bf34820)
herr_t H5VLfile_specific(void *obj, hid_t connector_id, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLdataset_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a43cee299f5913ddee33cd3a855da9048)
void * H5VLdataset_open(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t dapl_id, hid_t dxpl_id, void **req)
[H5VLconnector_str_to_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a458256651d397c69a113dd180f50411f)
herr_t H5VLconnector_str_to_info(const char *str, hid_t connector_id, void **info)
[H5VLterminate](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a45e9fa8c9a6c037aed0b0521bb884148)
herr_t H5VLterminate(hid_t connector_id)
[H5VLblob_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4824cecde7ca9cdcee9df2f0db3c288a)
herr_t H5VLblob_specific(void *obj, hid_t connector_id, void *blob_id, H5VL_blob_specific_args_t *args)
[H5VLtoken_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a497f2cf9aefaba6f335be5a32dc3e109)
herr_t H5VLtoken_to_str(void *obj, H5I_type_t obj_type, hid_t connector_id, const H5O_token_t *token, char **token_str)
[H5VLdatatype_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a49f02bbb1985181993a530f109b33707)
herr_t H5VLdatatype_close(void *dt, hid_t connector_id, hid_t dxpl_id, void **req)
[H5VLget_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4ad793093a03375e7af24f27bc60c2e5)
herr_t H5VLget_cap_flags(hid_t connector_id, uint64_t *cap_flags)
[H5VLrequest_free](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4bdb448184824a25bb3f7ccecd99444b)
herr_t H5VLrequest_free(void *req, hid_t connector_id)
[H5VLlink_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a4d8d57fd847725b7f25817d2613858d7)
herr_t H5VLlink_specific(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLdataset_write](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a5028517e60ff4ae4a34a6c9ff1185668)
herr_t H5VLdataset_write(size_t count, void *dset[], hid_t connector_id, hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[], hid_t plist_id, const void *buf[], void **req)
[H5VLget_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a542800ebb029002c3f82f4efe2d50ee3)
void * H5VLget_object(void *obj, hid_t connector_id)
[H5VLrequest_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a56983158d53c6f71ee5c7613f96265ef)
herr_t H5VLrequest_specific(void *req, hid_t connector_id, H5VL_request_specific_args_t *args)
[H5VLfile_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a5c60fb975ab92946e798c29d4490bc33)
void * H5VLfile_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req)
[H5VLunwrap_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a613986d72333d30e5487ae334c8df0a1)
void * H5VLunwrap_object(void *obj, hid_t connector_id)
[H5VLdataset_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a678c02d9005a68920ca71c8078748fb5)
herr_t H5VLdataset_close(void *dset, hid_t connector_id, hid_t dxpl_id, void **req)
[H5VLdatatype_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6d9af2589e98fb0bc536a6dd1bc36ab1)
herr_t H5VLdatatype_specific(void *obj, hid_t connector_id, H5VL_datatype_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLgroup_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6e872a3063e66fae44331ad427c11f28)
void * H5VLgroup_create(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req)
[H5VLattr_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a6f5ab5c95feb563669c61e71a1af79c9)
herr_t H5VLattr_close(void *attr, hid_t connector_id, hid_t dxpl_id, void **req)
[H5VLrequest_wait](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a7c04eeb53490737229ec93252b86e2f2)
herr_t H5VLrequest_wait(void *req, hid_t connector_id, uint64_t timeout, H5VL_request_status_t *status)
[H5VLdataset_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a7cd52e5b61d504e7d6e3a769534efdc7)
herr_t H5VLdataset_get(void *dset, hid_t connector_id, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLdatatype_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a83e32eb893ad112df89d304bb32c3cea)
void * H5VLdatatype_open(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t tapl_id, hid_t dxpl_id, void **req)
[H5VLinitialize](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a8e43a9640d599a68b2ce281796920d88)
herr_t H5VLinitialize(hid_t connector_id, hid_t vipl_id)
[H5VLgroup_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a8f16a678c422196f5b269fcf64eb0f57)
herr_t H5VLgroup_get(void *obj, hid_t connector_id, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLcopy_connector_info](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a910252aef3ceccad24ccc1cd03a38450)
herr_t H5VLcopy_connector_info(hid_t connector_id, void **dst_vol_info, void *src_vol_info)
[H5VLtoken_cmp](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a91cd262c9fced15807636937f8ae91b6)
herr_t H5VLtoken_cmp(void *obj, hid_t connector_id, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value)
[H5VLobject_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a91f97ca5ec160ed4f2bff2fc1949cddd)
herr_t H5VLobject_specific(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLdataset_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9311dc565e5286c9f6e7d6594cf55781)
herr_t H5VLdataset_optional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLrequest_notify](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9b490e7864366df596270c003af74468)
herr_t H5VLrequest_notify(void *req, hid_t connector_id, H5VL_request_notify_t cb, void *ctx)
[H5VLdataset_read](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#a9e8670dd258b866391da0633a823a01d)
herr_t H5VLdataset_read(size_t count, void *dset[], hid_t connector_id, hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[], hid_t plist_id, void *buf[], void **req)
[H5VLlink_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aa493c236d15cd60a610fbb4821e8d095)
herr_t H5VLlink_optional(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLgroup_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aaa255508c8fd64a8f24ab66497ac3800)
herr_t H5VLgroup_specific(void *obj, hid_t connector_id, H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLconnector_info_to_str](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aac2b19b03066f3f9e07aae264de6bd14)
herr_t H5VLconnector_info_to_str(const void *info, hid_t connector_id, char **str)
[H5VLwrap_object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aac5db40d46b03bf31d9cee91fdb5ca14)
void * H5VLwrap_object(void *obj, H5I_type_t obj_type, hid_t connector_id, void *wrap_ctx)
[H5VLdatatype_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ab0de23d096ca426c2c7c81c22ca6ccf7)
herr_t H5VLdatatype_optional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLdataset_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ab2b71f560cd6112c4b07de94335b30e8)
void * H5VLdataset_create(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req)
[H5VLfile_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#abedac242a6b05c74630aee6717c6eb86)
void * H5VLfile_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req)
[H5VLdatatype_commit](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac3f34096fc3f8a0e939eb479bdbd1f38)
void * H5VLdatatype_commit(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id, hid_t dxpl_id, void **req)
[H5VLattr_write](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac40f6f460fe5bc4de1db2f5ee7bdc647)
herr_t H5VLattr_write(void *attr, hid_t connector_id, hid_t dtype_id, const void *buf, hid_t dxpl_id, void **req)
[H5VLattr_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac632a877cce1103f4e599959d4cc5460)
herr_t H5VLattr_specific(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req)
[H5VLlink_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ac87aec9385917d6f0d572de479f0bdb6)
herr_t H5VLlink_get(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, H5VL_link_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLgroup_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#acd34999e825ada4964f0d4d367cad42f)
void * H5VLgroup_open(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t gapl_id, hid_t dxpl_id, void **req)
[H5VLobject_copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad3e5984fa0bebddfe9d38e8ba0257ceb)
herr_t H5VLobject_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, const char *src_name, void *dst_obj, const H5VL_loc_params_t *loc_params2, const char *dst_name, hid_t connector_id, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req)
[H5VLget_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad4d06542aef57546ccd1e40c4955e03e)
herr_t H5VLget_wrap_ctx(void *obj, hid_t connector_id, void **wrap_ctx)
[H5VLattr_read](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ad5a4f0da0e12b4f37df203c982687bdb)
herr_t H5VLattr_read(void *attr, hid_t connector_id, hid_t dtype_id, void *buf, hid_t dxpl_id, void **req)
[H5VLdatatype_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#adafa64999e4dda2540843fe333a6a884)
herr_t H5VLdatatype_get(void *dt, hid_t connector_id, H5VL_datatype_get_args_t *args, hid_t dxpl_id, void **req)
[H5VLget_value](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#adc2fe50b5e8945e1ba778c05118381f2)
herr_t H5VLget_value(hid_t connector_id, H5VL_class_value_t *conn_value)
[H5VLrequest_cancel](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae125d77504bfc8ba5edf68a5d1796856)
herr_t H5VLrequest_cancel(void *req, hid_t connector_id, H5VL_request_status_t *status)
[H5VLattr_create](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae52e1eebbdd8dc7ceb179ab694552a98)
void * H5VLattr_create(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *attr_name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req)
[H5VLattr_open](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae795c2b22a3c8c610770332082d5d567)
void * H5VLattr_open(void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id, const char *name, hid_t aapl_id, hid_t dxpl_id, void **req)
[H5VLfile_optional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#ae9801ec2575976a0b5ed461ecb0b3689)
herr_t H5VLfile_optional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLblob_get](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aecf782e83423e79fa4660c0d1b1978a8)
herr_t H5VLblob_get(void *obj, hid_t connector_id, const void *blob_id, void *buf, size_t size, void *ctx)
[H5VLoptional](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aeeacac4f0290962703435fd2f4b794be)
herr_t H5VLoptional(void *obj, hid_t connector_id, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
[H5VLfree_wrap_ctx](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#af769237a892ee3c20d7cd75d22a64fc6)
herr_t H5VLfree_wrap_ctx(void *wrap_ctx, hid_t connector_id)
[H5VLgroup_close](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#afa20af338f0722587bec9af00e7a041c)
herr_t H5VLgroup_close(void *grp, hid_t connector_id, hid_t dxpl_id, void **req)
[H5VLdataset_specific](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_lconnector__passthru_8h.html#aff9b695f422d86e9aff84a165acd2658)
herr_t H5VLdataset_specific(void *obj, hid_t connector_id, H5VL_dataset_specific_args_t *args, hid_t dxpl_id, void **req)
#  Appendix C Native VOL Connector Optional Values By Subclass
/* H5VL_SUBCLS_ATTR */
#define H5VL_NATIVE_ATTR_ITERATE_OLD 0 /* H5Aiterate (deprecated routine) */
/* H5VL_SUBCLS_DATASET */
#define H5VL_NATIVE_DATASET_FORMAT_CONVERT 0 /* H5Dformat_convert (internal) */
#define H5VL_NATIVE_DATASET_GET_CHUNK_INDEX_TYPE 1 /* H5Dget_chunk_index_type */
#define H5VL_NATIVE_DATASET_GET_CHUNK_STORAGE_SIZE 2 /* H5Dget_chunk_storage_size */
#define H5VL_NATIVE_DATASET_GET_NUM_CHUNKS 3 /* H5Dget_num_chunks */
#define H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_IDX 4 /* H5Dget_chunk_info */
#define H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_COORD 5 /* H5Dget_chunk_info_by_coord */
#define H5VL_NATIVE_DATASET_CHUNK_READ 6 /* H5Dchunk_read */
#define H5VL_NATIVE_DATASET_CHUNK_WRITE 7 /* H5Dchunk_write */
#define H5VL_NATIVE_DATASET_GET_VLEN_BUF_SIZE 8 /* H5Dvlen_get_buf_size */
#define H5VL_NATIVE_DATASET_GET_OFFSET 9 /* H5Dget_offset */
#define H5VL_NATIVE_DATASET_CHUNK_ITER 10 /* H5Dchunk_iter */
/* H5VL_SUBCLS_FILE */
#define H5VL_NATIVE_FILE_CLEAR_ELINK_CACHE 0 /* H5Fclear_elink_file_cache */
#define H5VL_NATIVE_FILE_GET_FILE_IMAGE 1 /* H5Fget_file_image */
#define H5VL_NATIVE_FILE_GET_FREE_SECTIONS 2 /* H5Fget_free_sections */
#define H5VL_NATIVE_FILE_GET_FREE_SPACE 3 /* H5Fget_freespace */
#define H5VL_NATIVE_FILE_GET_INFO 4 /* H5Fget_info1/2 */
#define H5VL_NATIVE_FILE_GET_MDC_CONF 5 /* H5Fget_mdc_config */
#define H5VL_NATIVE_FILE_GET_MDC_HR 6 /* H5Fget_mdc_hit_rate */
#define H5VL_NATIVE_FILE_GET_MDC_SIZE 7 /* H5Fget_mdc_size */
#define H5VL_NATIVE_FILE_GET_SIZE 8 /* H5Fget_filesize */
#define H5VL_NATIVE_FILE_GET_VFD_HANDLE 9 /* H5Fget_vfd_handle */
#define H5VL_NATIVE_FILE_RESET_MDC_HIT_RATE 10 /* H5Freset_mdc_hit_rate_stats */
#define H5VL_NATIVE_FILE_SET_MDC_CONFIG 11 /* H5Fset_mdc_config */
#define H5VL_NATIVE_FILE_GET_METADATA_READ_RETRY_INFO 12 /* H5Fget_metadata_read_retry_info */
#define H5VL_NATIVE_FILE_START_SWMR_WRITE 13 /* H5Fstart_swmr_write */
#define H5VL_NATIVE_FILE_START_MDC_LOGGING 14 /* H5Fstart_mdc_logging */
#define H5VL_NATIVE_FILE_STOP_MDC_LOGGING 15 /* H5Fstop_mdc_logging */
#define H5VL_NATIVE_FILE_GET_MDC_LOGGING_STATUS 16 /* H5Fget_mdc_logging_status */
#define H5VL_NATIVE_FILE_FORMAT_CONVERT 17 /* H5Fformat_convert */
#define H5VL_NATIVE_FILE_RESET_PAGE_BUFFERING_STATS 18 /* H5Freset_page_buffering_stats */
#define H5VL_NATIVE_FILE_GET_PAGE_BUFFERING_STATS 19 /* H5Fget_page_buffering_stats */
#define H5VL_NATIVE_FILE_GET_MDC_IMAGE_INFO 20 /* H5Fget_mdc_image_info */
#define H5VL_NATIVE_FILE_GET_EOA 21 /* H5Fget_eoa */
#define H5VL_NATIVE_FILE_INCR_FILESIZE 22 /* H5Fincrement_filesize */
#define H5VL_NATIVE_FILE_SET_LIBVER_BOUNDS 23 /* H5Fset_latest_format/libver_bounds */
#define H5VL_NATIVE_FILE_GET_MIN_DSET_OHDR_FLAG 24 /* H5Fget_dset_no_attrs_hint */
#define H5VL_NATIVE_FILE_SET_MIN_DSET_OHDR_FLAG 25 /* H5Fset_dset_no_attrs_hint */
#ifdef H5_HAVE_PARALLEL
#define H5VL_NATIVE_FILE_GET_MPI_ATOMICITY 26 /* H5Fget_mpi_atomicity */
#define H5VL_NATIVE_FILE_SET_MPI_ATOMICITY 27 /* H5Fset_mpi_atomicity */
#endif
#define H5VL_NATIVE_FILE_POST_OPEN 28 /* Adjust file after open, with wrapping context */
/* H5VL_SUBCLS_GROUP */
#define H5VL_NATIVE_GROUP_ITERATE_OLD 0 /* HG5Giterate (deprecated routine) */
#define H5VL_NATIVE_GROUP_GET_OBJINFO 1 /* HG5Gget_objinfo (deprecated routine) */
/* H5VL_SUBCLS_OBJECT */
#define H5VL_NATIVE_OBJECT_GET_COMMENT 0 /* H5G|H5Oget_comment, H5Oget_comment_by_name */
#define H5VL_NATIVE_OBJECT_SET_COMMENT 1 /* H5G|H5Oset_comment, H5Oset_comment_by_name */
#define H5VL_NATIVE_OBJECT_DISABLE_MDC_FLUSHES 2 /* H5Odisable_mdc_flushes */
#define H5VL_NATIVE_OBJECT_ENABLE_MDC_FLUSHES 3 /* H5Oenable_mdc_flushes */
#define H5VL_NATIVE_OBJECT_ARE_MDC_FLUSHES_DISABLED 4 /* H5Oare_mdc_flushes_disabled */
#define H5VL_NATIVE_OBJECT_GET_NATIVE_INFO 5 /* H5Oget_native_info(_by_idx, _by_name) */
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


