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
  * [ Obtain Tools and Files (Optional)](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_command.html#title0 "H1")
  * [ Tutorial Topics](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_command.html#title1 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Command-line Tools
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
* * *
#  Obtain Tools and Files (Optional)
Pre-built binaries for Linux and Windows are distributed within the respective HDF5 binary release packages, which can be obtained from the [Download HDF5](https://support.hdfgroup.org/releases/hdf5/downloads/latest/index.html) page.
HDF5 files can be obtained from various places such as [HDF5 Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h_d_f5_examples.html) and [HDF-EOS and Tools and Information Center](http://www.hdfeos.org/). Specifically, the following examples are used in this tutorial topic: 
  * HDF5 Files created from compiling the [Examples from Learning the Basics](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_examples.html)
  * HDF5 Files on the [Examples by API](https://support.hdfgroup.org/documentation/hdf5/latest/_ex_a_p_i.html) page 
  * NPP JPSS files, [SVM01_npp.. (gzipped)](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/SVM01_npp_d20130524_t1255132_e1256374_b08146_c20130524192048864992_noaa_ops.h5.gz) and [SVM09_npp.. (gzipped)](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/SVM09_npp_d20120229_t0849107_e0854511_b01759_c20120229145452682127_noaa_ops.h5.gz)
  * HDF-EOS [OMI-Aura file](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/OMI-Aura.he5)


#  Tutorial Topics
A variety of command-line tools are included in the HDF5 binary distribution. There are tools to view, edit, convert and compare HDF5 files. This tutorial discusses the tools by their functionality. It does not cover all of the HDF5 tools.
Tool Category  | Topic  | Tools Used   
---|---|---  
**[Command-line Tools For Viewing HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html)** |  [File Content and Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewContent) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)  
|  [Datasets and Dataset Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewDset) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)  
|  [Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewGrps) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)  
|  [Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewAttr) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)  
|  [Dataset Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewSub) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)  
|  [Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewDtypes) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)  
[Command-line Tools For Editing HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html) |  [Remove Inaccessible Objects and Unused Space in a File](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html#secViewToolsEditRemove) |  [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack)  
|  [Change a Dataset's Storage Layout](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html#secViewToolsEditChange) |  [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack)  
|  [Apply Compression Filter to a Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html#secViewToolsEditApply) |  [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html#sec_cltools_h5repack)  
|  [Copy Objects to Another File](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html#secViewToolsEditCopy) |  [h5copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html#sec_cltools_h5copy)  
|  [Add or Remove User Block from File](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_edit.html#secViewToolsEditAdd) |  [h5jam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#sec_cltools_h5jam) and [h5unjam](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__j_a_m__u_g.html#sec_cltools_h5unjam)  
[Command-line Tools For Converting HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_convert.html) |  [Output HDF5 Dataset into an ASCII File (to Import into Excel and Other Applications)](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_convert.html#secViewToolsConvertASCII) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)  
|  [Output HDF5 Dataset into Binary File](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_convert.html#secViewToolsConvertBinary) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)  
|  [Export from h5dump and Import into HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_convert.html#secViewToolsConvertExport) |  [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5import](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__i_m__u_g.html#sec_cltools_h5import)  
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


