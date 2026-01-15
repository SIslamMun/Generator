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


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Reference Manual
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
* * *
The functions provided by the HDF5 API are grouped into the following _modules_ :
Modules   
---  
Core Reference Manual Modules | Module | Language | Description   
---|---|---  
Attributes (H5A) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_a.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_a.html) | HDF5 attribute is a small metadata object describing the nature and/or intended usage of a primary data object.   
Datasets (H5D) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_d.html) | Manage HDF5 datasets, including the transfer of data between memory and disk and the description of dataset properties.   
Dataspaces (H5S) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_s.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html) | HDF5 dataspaces describe the shape of datasets in memory or in HDF5 files.   
Datatypes (H5T) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_t.html) | HDF5 datatypes describe the element type of HDF5 datasets and attributes.   
Error Handling (H5E) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_e.html) | HDF5 library error reporting.   
Event Set (H5ES) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html) | Java | HDF5 event set life cycle used with HDF5 VOL connectors that enable the asynchronous feature in HDF5.   
Files (H5F) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_file.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_f.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_f.html) | Manage HDF5 files.   
Filters (H5Z) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_z.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_z.html) | Manage HDF5 user-defined filters.   
Groups (H5G) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_g.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html) | Manage HDF5 groups.   
Identifiers (H5I) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_i.html) | Manage identifiers defined by the HDF5 library.   
Library General (H5) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5.html) | Manage the life cycle of HDF5 library instances.   
Links (H5L) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html) | Manage HDF5 links and link types.   
Objects (H5O) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_o.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_o.html) | Manage HDF5 objects (groups, datasets, datatype objects).   
Property Lists (H5P) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) | [C++](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html) | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_p.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p.html) | HDF5 property lists are the main vehicle to configure the behavior of HDF5 API functions.   
Dynamically-loaded Plugins (H5PL) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_l.html) | C++ | Fortran | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_p_l.html) | Manage the loading behavior of HDF5 plugins.   
References (H5R) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_r.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_r.html) | Manage HDF5 references (HDF5 objects, attributes, and selections on datasets a.k.a. dataset regions).   
Virtual File Driver (H5FD) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_f_d.html) | Java | Manage HDF5 Virtual File Driver.   
VOL Connector (H5VL) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_v_l.html) | [Java](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_v_l.html) | Manage HDF5 VOL connector plugins.   
High-level Reference Manual Modules Module | Language | Description   
---|---|---  
HDF5 Lite APIs (H5LT,H5LD) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_l_t.html) | Java | Functions to simplify creating and manipulating datasets, attributes and other features.   
HDF5 Images API (H5IM) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_i_m.html) | Java | Creating and manipulating HDF5 datasets intended to be interpreted as images.   
HDF5 Table APIs (H5TB) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html) | Java | Creating and manipulating HDF5 datasets intended to be interpreted as tables.   
HDF5 Packet Table APIs (H5PT) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html) | C++ | Fortran | Java | Creating and manipulating HDF5 datasets to support append- and read-only operations on table data.   
HDF5 Dimension Scales APIs (H5DS) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_s.html) | Java | Creating and manipulating HDF5 datasets that are associated with the dimension of another HDF5 dataset.   
HDF5 Optimizations APIs (H5DO) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html) | C++ | [Fortran](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d_o.html) | Java | Bypassing default HDF5 behavior in order to optimize for specific use cases.   
Extensions (H5LR, H5LT) | [C](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html) | C++ | Fortran | Java |   
Additional Java Reference Manual Modules | [Constants and Enumerated Types](https://support.hdfgroup.org/documentation/hdf5/latest/_h_d_f5_c_o_n_s_t.html) | This class contains C constants and enumerated types of HDF5 library.   
---|---  
[Native Arrays of Numbers](https://support.hdfgroup.org/documentation/hdf5/latest/_h_d_f_n_a_t_i_v_e.html) | This class encapsulates native methods to deal with arrays of numbers, converting from numbers to bytes and bytes to numbers.   
[Java Array Conversion](https://support.hdfgroup.org/documentation/hdf5/latest/_h_d_f_a_r_r_a_y.html) | This is a class for handling multidimensional arrays for HDF.   
[Errors and Exceptions](https://support.hdfgroup.org/documentation/hdf5/latest/_e_r_r_o_r_s.html) | The class HDF5Exception returns errors from the Java HDF5 Interface.   
[HDF5 Predefined Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/predefined_datatypes_tables.html)  
[Deprecated List](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html)  
Functions with [Asynchronous Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html)  
[API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)  
Follow these simple rules and stay out of trouble:
  * **Handle discipline:** The HDF5 API is rife with handles or identifiers, which you typically obtain by creating new HDF5 items, copying items, or retrieving facets of items. Consequently, **and most importantly** , you are responsible for releasing the underlying resources via the matching `H5*close()` call, or deal with the consequences of resource leakage. 
  * **Closed means closed:** Do not pass identifiers that were previously `H5*close()`-d to other API functions! It will generate an error. 
  * **Dynamic memory allocation:** The API contains a few functions in which the HDF5 library dynamically allocates memory on the caller's behalf. The caller owns this memory and eventually must free it by calling [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.") and not language-explicit memory functions. 
  * **Don't modify while iterating:** Do not modify the underlying collection when an iteration is in progress! 
  * **Use of locations:** Certain API functions, typically called `H5***_by_name` use a combination of identifiers and path names to refer to HDF5 objects. If the identifier fully specifies the object in question, pass `'.'` (a dot) for the name!



Attention
     **C++ Developers using HDF5 C-API functions beware:**  
Several functions in this C-API take function pointers or callbacks as arguments. Examples include [H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list."), [H5Pset_type_conv_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga10a80b29444d933da1aa2003f46cf003 "Sets user-defined datatype conversion callback function."), [H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another."), and [H5Ewalk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4ecc0f6a1ea5bb821373a5a7b8070655 "Walks the specified error stack, calling the specified function."). Application code must ensure that those callback functions return normally such to allow the HDF5 to manage its resources and maintain a consistent state. For instance, those functions must not use the C `setjmp` / `longjmp` mechanism to leave those callback functions. Within the context of C++, any exceptions thrown within the callback function must be caught, such as with a `catch(…)` statement. Any exception state can be placed within the provided user data function call arguments, and may be thrown again once the calling function has returned. Exceptions raised and not handled inside the callback are not supported as it might leave the HDF5 library in an inconsistent state. Similarly, using C++20 coroutines cannot be used as callbacks, since they do not support plain return statements. If a callback function yields execution to another C++20 coroutine calling HDF5 functions as well, this may lead to undefined behavior. 

C++ API Guide
    New to the HDF5 C++ API? See the [C++ API Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html) for an introduction, class mapping guide, and examples. 

Don't like what you see? - You can help to improve this Reference Manual
    Complete the survey linked near the top of this page!  
We treat documentation like code: Fork the [HDF5 repo](https://github.com/HDFGroup/hdf5), make changes, and create a [pull request](https://github.com/HDFGroup/hdf5/pulls) !  
See the [Reference Manual (RM) Page Template](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m_t.html) for general guidance.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


