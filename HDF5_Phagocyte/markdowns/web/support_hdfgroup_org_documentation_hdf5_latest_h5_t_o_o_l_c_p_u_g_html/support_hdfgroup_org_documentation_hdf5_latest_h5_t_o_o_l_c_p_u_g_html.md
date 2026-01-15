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
  * [ h5copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title1 "H2")
  * [ Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title2 "H2")
  * [ Objects](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title3 "H2")
  * [ Error Report Option](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title4 "H2")
  * [ Options](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
The HDF5 h5copy Tool
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
* * *
#  h5copy
##  Introduction
With h5copy, you can copy objects from an HDF5 file to another file.
##  Usage
#### h5copy [OPTIONS] [OBJECTS...]
##  Objects
  * **–input** input file name 
  * **–output** output file name 
  * **–source** source object name 
  * **–destination** destination object name


##  Error Report Option
  * **–enable-error-stack** Prints messages from the HDF5 error stack as they occur. Optional value 2 also prints file open errors, –enable-error-stack=2.


##  Options
  * **–help** Print a usage message and exit 
  * **–parents** No error if existing, make parent groups as needed 
  * **–verbose** Print information about OBJECTS and OPTIONS 
  * **–version** Print the library version number and exit 
  * **–flag** Flag type


###  Flag Type Options
Flag type is one of the following strings: 
  * **shallow** Copy only immediate members for groups 
  * **soft** Expand soft links into new objects 
  * **ext** Expand external links into new objects 
  * **ref** Copy references and any referenced objects, i.e., objects that the references point to.  
Referenced objects are copied in addition to the objects specified on the command line and reference datasets are populated with correct reference values. Copies of referenced datasets outside the copy range specified on the command line will normally have a different name from the original.  
(Default: Without this option, reference value(s) in any reference datasets are set to NULL and referenced objects are not copied unless they are otherwise within the copy range specified on the command line.) 
  * **noattr** Copy object without copying attributes 
  * **allflags** Switches all flags from the default to the non-default setting


These flag types correspond to the following API symbols 
  * **[H5O_COPY_SHALLOW_HIERARCHY_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#af17242deb8f10196a580daf25daa17a0)**
  * **[H5O_COPY_EXPAND_SOFT_LINK_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a649171ae36665d655d9f13906462c736)**
  * **[H5O_COPY_EXPAND_EXT_LINK_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a52b583ea3c1dd9cbc49a8c36a0383ca6)**
  * **[H5O_COPY_EXPAND_REFERENCE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a8e951bb94b78613a9f4d90b53287f024)**
  * **[H5O_COPY_WITHOUT_ATTR_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#abc54bab3b27a232d430a3e836506de4c)**
  * **[H5O_COPY_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a2b399814e8e4dd25f62c3a69db74fc4f)**


Next Chapter [h5diff](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_f__u_g.html#sec_cltools_h5diff)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_command_tools.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


