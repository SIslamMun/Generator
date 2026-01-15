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
  * [ Overview](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html#title0 "H1")
  * [ Prerequisites](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html#title1 "H2")
  * [ C API to C++ Class Mapping](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html#title2 "H2")
  * [ Quick Start Examples](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html#title3 "H2")
  * [ Building Applications](https://support.hdfgroup.org/documentation/hdf5/latest/h5_cpp_intro.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
C++ API Introduction
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
* * *
#  Overview
The HDF5 C++ API provides object-oriented wrappers for the HDF5 C Library, enabling modern C++ applications to leverage HDF5's powerful data management capabilities with:
  * Automatic resource management through RAII
  * Exception-based error handling
  * Type-safe interfaces
  * STL integration (std::string support)


##  Prerequisites
Users should be familiar with:
  * HDF5 concepts (files, groups, datasets, attributes, datatypes, dataspaces)
  * Basic C++ programming (classes, exceptions, templates)
  * The HDF5 [Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) for underlying C API concepts


##  C API to C++ Class Mapping
The C++ classes closely mirror the HDF5 C API modules:
C API Module  | C++ Class  | Purpose   
---|---|---  
H5A (Attributes)  |  [H5::Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html "Class Attribute operates on HDF5 attributes.") | Metadata attached to objects   
H5D (Datasets)  |  [H5::DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html "Class DataSet operates on HDF5 datasets.") | Data storage and I/O   
H5S (Dataspaces)  |  [H5::DataSpace](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html "Class DataSpace inherits from IdComponent and provides wrappers for the HDF5's dataspaces.") | Dimensional structure of data   
H5T (Datatypes)  |  [H5::DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") | Data type definitions   
H5F (Files)  |  [H5::H5File](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_file.html "Class H5File represents an HDF5 file and inherits from class Group as file is a root group.") | File access and management   
H5G (Groups)  |  [H5::Group](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html "Class Group represents an HDF5 group.") | Hierarchical organization   
H5P (Property Lists)  |  [H5::PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") + subclasses  | Configuration parameters   
H5L/H5O (Links/Objects)  | H5::Location  | Management of links and objects   
H5E (Errors)  |  [H5::Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html "Exception provides wrappers of HDF5 error handling functions.") | Error reporting   
##  Quick Start Examples
  * Creating groups and datasets: [h5tutr_crtgrp.cpp](https://\\CPPEXS/h5tutr_crtgrp.cpp) [h5tutr_crtdat.cpp](https://\\CPPEXS/h5tutr_crtdat.cpp)
  * Writing/Reading data [h5tutr_rdwt.cpp](https://\\CPPEXS/h5tutr_rdwt.cpp) [h5tutr_subset.cpp](https://\\CPPEXS/h5tutr_subset.cpp)
  * Working with attributes [h5tutr_crtatt.cpp](https://\\CPPEXS/h5tutr_crtatt.cpp)
  * Extendible datasets [h5tutr_extend.cpp](https://\\CPPEXS/h5tutr_extend.cpp)
  * Chunked/Compressed datasets [h5tutr_cmprss.cpp](https://\\CPPEXS/h5tutr_cmprss.cpp)


See the See [Examples from Learning the Basics](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_examples.html) for additional examples.
##  Building Applications
Please refer to the release_docs/INSTALL_CMake.txt file under the top directory of the HDF5 source code or [INSTALL_CMake.txt on GitHub for information about installing, building, and testing the C++ API.](https://github.com/HDFGroup/hdf5/blob/develop/release_docs/INSTALL_CMake.txt)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


