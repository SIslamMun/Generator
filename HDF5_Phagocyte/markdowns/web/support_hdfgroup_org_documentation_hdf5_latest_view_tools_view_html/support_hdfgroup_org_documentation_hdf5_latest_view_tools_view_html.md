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
  * [ Contents](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title0 "H1")
  * [ File Content and Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title1 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title2 "H2")
  * [ h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title3 "H2")
  * [ Datasets and Dataset Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title4 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title5 "H2")
  * [ h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title6 "H2")
  * [ Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title7 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title8 "H2")
  * [ h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title9 "H2")
  * [ Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title10 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title11 "H2")
  * [ h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title12 "H2")
  * [ Dataset Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title13 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title14 "H2")
  * [ Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title15 "H1")
  * [ h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#title16 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Command-line Tools For Viewing HDF5 Files
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html) / [Command-line Tools](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_command.html)
* * *
#  Contents
  * [File Content and Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewContent)
  * [Datasets and Dataset Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewDset)
  * [Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewGrps)
  * [Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewAttr)
  * [Dataset Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewSub)
  * [Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewDtypes)


#  File Content and Structure
The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) tools can both be used to view the contents of an HDF5 file. The tools are discussed below: 
  * [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewContent_h5dump)
  * [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewContent_h5ls)


##  h5dump
The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) tool dumps or displays the contents of an HDF5 file (textually). By default if you specify no options, [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) will display the entire contents of a file. There are many [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options for examining specific details of a file. To see all of the available [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options, specify the `-h` or `–help` option: 
h5dump -h
The following [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options can be helpful in viewing the content and structure of a file: 
Option  | Description  | Comment   
---|---|---  
-n, –contents  | Displays a list of the objects in a file  | See [Example 1](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewContent_h5dumpEx1)  
-n 1, –contents=1  | Displays a list of the objects and attributes in a file  | See [Example 6](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewAttr_h5dumpEx6)  
-H, –header  | Displays header information only (no data)  | See [Example 2](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewContent_h5dumpEx2)  
-A 0, –onlyattr=0  | Suppresses the display of attributes  | See [Example 2](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewContent_h5dumpEx2)  
-N P, –any_path=P  | Displays any object or attribute that matches path P  | See [Example 6](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewAttr_h5dumpEx6)  
###  Example 1
The following command displays a list of the objects in the file OMI-Aura.he5 (an HDF-EOS5 file): 
h5dump -n OMI-Aura.he5
As shown in the output below, the objects (groups, datasets) are listed to the left, followed by their names. You can see that this file contains two root groups, HDFEOS and HDFEOS INFORMATION: 
HDF5 "OMI-Aura.he5" {
FILE_CONTENTS {
group /
group /HDFEOS
group /HDFEOS/ADDITIONAL
group /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
group /HDFEOS/GRIDS
group /HDFEOS/GRIDS/OMI Column Amount O3
group /HDFEOS/GRIDS/OMI Column Amount O3/Data Fields
dataset /HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ColumnAmountO3
dataset /HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/RadiativeCloudFraction
dataset /HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle
dataset /HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ViewingZenithAngle
group /HDFEOS INFORMATION
dataset /HDFEOS INFORMATION/StructMetadata.0
}
}
###  Example 2
The file structure of the OMI-Aura.he5 file can be seen with the following command. The -A 0 option suppresses the display of attributes: 
h5dump -H -A 0 OMI-Aura.he5
Output of this command is shown below: 
HDF5 "OMI-Aura.he5" {
GROUP "/" {
GROUP "HDFEOS" {
GROUP "ADDITIONAL" {
GROUP "FILE_ATTRIBUTES" {
}
}
GROUP "GRIDS" {
GROUP "OMI Column Amount O3" {
GROUP "Data Fields" {
DATASET "ColumnAmountO3" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "RadiativeCloudFraction" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "SolarZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "ViewingZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
}
}
}
}
GROUP "HDFEOS INFORMATION" {
DATASET "StructMetadata.0" {
DATATYPE [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) {
STRSIZE 32000;
STRPAD [H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13);
CSET [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663);
CTYPE [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d);
}
DATASPACE SCALAR
}
}
}
}
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663)
@ H5T_CSET_ASCII
**Definition** H5Tpublic.h:96
[H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa)
@ H5T_STRING
**Definition** H5Tpublic.h:35
[H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13)
@ H5T_STR_NULLTERM
**Definition** H5Tpublic.h:121
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
#define H5T_IEEE_F32LE
**Definition** H5Tpublic.h:271
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d)
#define H5T_C_S1
**Definition** H5Tpublic.h:646 

See also
    H5TOOL_DP_UG for more information.
##  h5ls
The [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) tool by default just displays the objects in the root group. It will not display items in groups beneath the root group unless specified. Useful [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) options for viewing file content and structure are: 
Option  | Description  | Comment   
---|---|---  
-r  | Lists all groups and objects recursively  | See [Example 3](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewContent_h5lsEx3)  
-v  | Generates verbose output (lists dataset properties, attributes and attribute values, but no dataset values)  |   
###  Example 3
The following command shows the contents of the HDF-EOS5 file OMI-Aura.he5. The output is similar to [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump), except that [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) also shows dataspace information for each dataset: 
h5ls -r OMI-Aura.he5
The output is shown below: 
/ Group
/HDFEOS Group
/HDFEOS/ADDITIONAL Group
/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES Group
/HDFEOS/GRIDS Group
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3 Group
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields Group
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/ColumnAmountO3 Dataset {720, 1440}
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/RadiativeCloudFraction Dataset {720, 1440}
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/SolarZenithAngle Dataset {720, 1440}
/HDFEOS/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/ViewingZenithAngle Dataset {720, 1440}
/HDFEOS\ INFORMATION Group
/HDFEOS\ INFORMATION/StructMetadata.0 Dataset {SCALAR} 

See also
    H5TOOL_LS_UG for more information.
#  Datasets and Dataset Properties
Both [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and h5ls can be used to view specific datasets. 
  * [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewDset_h5dump)
  * [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewDset_h5ls)


##  h5dump
Useful [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options for examining specific datasets include: 
Option  | Description  | Comment   
---|---|---  
-d D, –dataset=D  | Displays dataset D  | See [Example 4](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDset_h5dumpEx4)  
-H, –header  | Displays header information only  | See [Example 4](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDset_h5dumpEx4)  
-p, –properties  | Displays dataset filters, storage layout, and fill value properties  | See [Example 5](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDset_h5dumpEx5)  
-A 0, –onlyattr=0  | Suppresses the display of attributes  | See [Example 2](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewContent_h5dumpEx2)  
-N P, –any_path=P  | Displays any object or attribute that matches path P  | See [Example 6](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewAttr_h5dumpEx6)  
###  Example 4
A specific dataset can be viewed with `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` using the `-d D` option and specifying the entire path and name of the dataset for `D`. The path is important in identifying the correct dataset, as there can be multiple datasets with the same name. The path can be determined by looking at the objects in the file with `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) -n`.
The following example uses the `groups.h5` file that is created by the [Examples from Learning the Basics](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_examples.html) example `h5_crtgrpar.c`. To display `dset1` in the `groups.h5` file below, specify dataset `/MyGroup/dset1`. The `-H` option is used to suppress printing of the data values:
_Contents of groups.h5_
$ h5dump -n groups.h5
HDF5 "groups.h5" {
FILE_CONTENTS {
group /
group /MyGroup
group /MyGroup/Group_A
dataset /MyGroup/Group_A/dset2
group /MyGroup/Group_B
dataset /MyGroup/dset1
}
}
_Display dataset "dset1"_
$ h5dump -d "/MyGroup/dset1" -H groups.h5
HDF5 "groups.h5" {
DATASET "/MyGroup/dset1" {
DATATYPE [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
DATASPACE SIMPLE { ( 3, 3 ) / ( 3, 3 ) }
}
}
[H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
#define H5T_STD_I32BE
**Definition** H5Tpublic.h:466
###  Example 5
The `-p` option is used to examine the dataset filters, storage layout, and fill value properties of a dataset.
This option can be useful for checking how well compression works, or even for analyzing performance and dataset size issues related to chunking. (The smaller the chunk size, the more chunks that HDF5 has to keep track of, which increases the size of the file and potentially affects performance.)
In the file shown below the dataset `/DS1` is both chunked and compressed: 
$ h5dump -H -p -d "/DS1" h5ex_d_gzip.h5
HDF5 "h5ex_d_gzip.h5" {
DATASET "/DS1" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 32, 64 ) / ( 32, 64 ) }
STORAGE_LAYOUT {
CHUNKED ( 4, 8 )
SIZE 5278 (1.552:1 COMPRESSION)
}
FILTERS {
COMPRESSION DEFLATE { LEVEL 9 }
}
FILLVALUE {
FILL_TIME [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
VALUE 0
}
ALLOCATION_TIME {
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
}
}
}
[H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
@ H5D_FILL_TIME_IFSET
**Definition** H5Dpublic.h:119
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
@ H5D_ALLOC_TIME_INCR
**Definition** H5Dpublic.h:94
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
#define H5T_STD_I32LE
**Definition** H5Tpublic.h:471
You can obtain the `h5ex_d_gzip.c` program that created this file, as well as the file created, from the [Examples by API](https://support.hdfgroup.org/documentation/hdf5/latest/_ex_a_p_i.html) page.
##  h5ls
Specific datasets can be specified with `h5ls[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)` by simply adding the dataset path and dataset after the file name. As an example, this command displays dataset `dset2` in the `groups.h5` file used in [Example 4](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDset_h5dumpEx4) : 
h5ls groups.h5/MyGroup/Group_A/dset2
Just the dataspace information gets displayed: 
dset2 Dataset {2, 10}
The following options can be used to see detailed information about a dataset. 
Option  | Description   
---|---  
-v, –verbose  | Generates verbose output (lists dataset properties, attributes and attribute values, but no dataset values)   
-d, –data  | Displays dataset values   
The output of using `-v` is shown below: 
$ h5ls -v groups.h5/MyGroup/Group_A/dset2
Opened "groups.h5" with sec2 driver.
dset2 Dataset {2/2, 10/10}
Location: 1:3840
Links: 1
Storage: 80 logical bytes, 80 allocated bytes, 100.00% utilization
Type: 32-bit big-endian integer
The output of using `-d` is shown below: 
$ h5ls -d groups.h5/MyGroup/Group_A/dset2
dset2 Dataset {2, 10}
Data:
(0,0) 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#  Groups
Both [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) can be used to view specific groups in a file. 
  * [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewGrps_h5dump)
  * [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsecViewToolsViewGrps_h5ls)


##  h5dump
The `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` options that are useful for examining groups are: 
Option  | Description   
---|---  
-g G, –group=G  | Displays group G and its members   
-H, –header  | Displays header information only   
-A 0, –onlyattr=0  | Suppresses the display of attributes   
To view the contents of the `HDFEOS` group in the OMI file mentioned previously, you can specify the path and name of the group as follows: 
h5dump -g "/HDFEOS" -H -A 0 OMI-Aura.he5
The `-A 0` option suppresses attributes and `-H` suppresses printing of data values: 
HDF5 "OMI-Aura.he5" {
GROUP "/HDFEOS" {
GROUP "ADDITIONAL" {
GROUP "FILE_ATTRIBUTES" {
}
}
GROUP "GRIDS" {
GROUP "OMI Column Amount O3" {
GROUP "Data Fields" {
DATASET "ColumnAmountO3" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "RadiativeCloudFraction" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "SolarZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
DATASET "ViewingZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
}
}
}
}
}
}
##  h5ls
You can view the contents of a group with `h5ls[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)`/ by specifying the group after the file name. To use `h5ls[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls)` to view the contents of the `/HDFEOS` group in the `OMI-Aura.he5` file, type: 
h5ls -r OMI-Aura.he5/HDFEOS
The output of this command is: 
/ADDITIONAL Group
/ADDITIONAL/FILE_ATTRIBUTES Group
/GRIDS Group
/GRIDS/OMI\ Column\ Amount\ O3 Group
/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields Group
/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/ColumnAmountO3 Dataset {720, 1440}
/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/RadiativeCloudFraction Dataset {720, 1440}
/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/SolarZenithAngle Dataset {720, 1440}
/GRIDS/OMI\ Column\ Amount\ O3/Data\ Fields/ViewingZenithAngle Dataset {720, 1440}
If you specify the `-v` option, you can also see the attributes and properties of the datasets.
#  Attributes
##  h5dump
Attributes are displayed by default if using `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)`. Some files contain many attributes, which can make it difficult to examine the objects in the file. Shown below are options that can help when using `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` to work with files that have attributes.
###  Example 6
The `-a` A option will display an attribute. However, the path to the attribute must be included when specifying this option. For example, to see the `ScaleFactor` attribute in the `OMI-Aura.he5` file, type: 
h5dump -a "/HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle/ScaleFactor" OMI-Aura.he5
This command displays: 
HDF5 "OMI-Aura.he5" {
ATTRIBUTE "ScaleFactor" {
DATATYPE [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
DATASPACE SIMPLE { ( 1 ) / ( 1 ) }
DATA {
(0): 1
}
}
}
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
#define H5T_IEEE_F64LE
**Definition** H5Tpublic.h:283
How can you determine the path to the attribute? This can be done by looking at the file contents with the `-n 1` option: 
h5dump -n 1 OMI-Aura.he5
Below is a portion of the output for this command: 
HDF5 "OMI-Aura.he5" {
FILE_CONTENTS {
group /
group /HDFEOS
group /HDFEOS/ADDITIONAL
group /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/EndUTC
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/GranuleDay
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/GranuleDayOfYear
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/GranuleMonth
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/GranuleYear
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/InstrumentName
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/OrbitNumber
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/OrbitPeriod
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/PGEVersion
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/Period
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProcessLevel
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StartUTC
attribute /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/TAI93At0zOfGranule
...
There can be multiple objects or attributes with the same name in a file. How can you make sure you are finding the correct object or attribute? You can first determine how many attributes there are with a specified name, and then examine the paths to them.
The `-N` option can be used to display all objects or attributes with a given name. For example, there are four attributes with the name `ScaleFactor` in the `OMI-Aura.he5` file, as can be seen below with the `-N` option: 
h5dump -N ScaleFactor OMI-Aura.he5
It outputs: 
HDF5 "OMI-Aura.he5" {
ATTRIBUTE "ScaleFactor" {
DATATYPE [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
DATASPACE SIMPLE { ( 1 ) / ( 1 ) }
DATA {
(0): 1
}
}
ATTRIBUTE "ScaleFactor" {
DATATYPE [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
DATASPACE SIMPLE { ( 1 ) / ( 1 ) }
DATA {
(0): 1
}
}
ATTRIBUTE "ScaleFactor" {
DATATYPE [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
DATASPACE SIMPLE { ( 1 ) / ( 1 ) }
DATA {
(0): 1
}
}
ATTRIBUTE "ScaleFactor" {
DATATYPE [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
DATASPACE SIMPLE { ( 1 ) / ( 1 ) }
DATA {
(0): 1
}
}
}
##  h5ls
If you include the `-v` (verbose) option for `h5ls`, you will see all of the attributes for the specified file, dataset or group. You cannot display individual attributes.
#  Dataset Subset
##  h5dump
If you have a very large dataset, you may wish to subset or see just a portion of the dataset. This can be done with the following `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` options. 
Option  | Description   
---|---  
-d D, –dataset=D  | Dataset D   
-s START, –start=START  | Offset or start of subsetting selection   
-S STRIDE, –stride=STRIDE  | Stride (sampling along a dimension). The default (unspecified, or 1) selects every element along a dimension, a value of 2 selects every other element, a value of 3 selects every third element, ...   
-c COUNT, –count=COUNT  | Number of blocks to include in the selection   
-k BLOCK, –block=BLOCK  | Size of the block in a hyperslab. The default (unspecified, or 1) is for the block size to be the size of a single element.   
The `START (s)`, `STRIDE (S)`, `COUNT (c)`, and `BLOCK (k)` options define the shape and size of the selection. They are arrays with the same number of dimensions as the rank of the dataset's dataspace, and they all work together to define the selection. A change to one of these arrays can affect the others.
When specifying these [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options, a comma is used as the delimiter for each dimension in the option value. For example, with a 2-dimensional dataset, the option value is specified as "H,W", where H is the height and W is the width. If the offset is 0 for both dimensions, then `START` would be specified as follows: 
-s "0,0"
There is also a shorthand way to specify these options with brackets at the end of the dataset name: 
-d DATASETNAME[s;S;c;k]
Multiple dimensions are separated by commas. For example, a subset for a 2-dimensional dataset would be specified as follows: 
-d DATASETNAME[s,s;S,S;c,c;k,k]
For a detailed understanding of how selections works, see the [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.") API in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
The dataset SolarZenithAngle in the OMI-Aura.he5 file can be used to illustrate these options. This dataset is a 2-dimensional dataset of size 720 (height) x 1440 (width). Too much data will be displayed by simply viewing the specified dataset with the `-d` option: 
h5dump -d "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle" OMI-Aura.he5
Subsetting narrows down the output that is displayed. In the following example, the first 15x10 elements (-c "15,10") are specified, beginning with position (0,0) (-s "0,0"): 
h5dump -A 0 -d "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle" -s "0,0" -c "15,10" -w 0 OMI-Aura.he5
If using the shorthand method, specify: 
h5dump -A 0 -d "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle[0,0;;15,10;]" -w 0 OMI-Aura.he5
Where, the `-d` option must be specified before subsetting options (if not using the shorthand method).
The `-A 0` option suppresses the printing of attributes.
The `-w 0` option sets the number of columns of output to the maximum allowed value (65535). This ensures that there are enough columns specified for displaying the data.
Either command displays: 
HDF5 "OMI-Aura.he5" {
DATASET "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
SUBSET {
START ( 0, 0 );
STRIDE ( 1, 1 );
COUNT ( 15, 10 );
BLOCK ( 1, 1 );
DATA {
(0,0): 79.403, 79.403, 79.403, 79.403, 79.403, 79.403, 79.403, 79.403, 79.403, 79.403,
(1,0): 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071,
(2,0): 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867,
(3,0): 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632,
(4,0): 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429,
(5,0): 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225,
(6,0): 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021,
(7,0): 77.715, 77.715, 77.715, 77.715, 77.715, 77.715, 77.715, 77.715, 77.715, 77.715,
(8,0): 77.511, 77.511, 77.511, 77.511, 77.511, 77.511, 77.511, 77.511, 77.511, 77.511,
(9,0): 77.658, 77.658, 77.658, 77.307, 77.307, 77.307, 77.307, 77.307, 77.307, 77.307,
(10,0): 77.556, 77.556, 77.556, 77.556, 77.556, 77.556, 77.556, 77.556, 77.102, 77.102,
(11,0): 78.408, 78.408, 78.408, 78.408, 78.408, 78.408, 78.408, 78.408, 77.102, 77.102,
(12,0): 76.34, 78.413, 78.413, 78.413, 78.413, 78.413, 78.413, 78.413, 78.413, 78.413,
(13,0): 78.107, 78.107, 78.107, 78.107, 78.107, 78.107, 78.107, 78.107, 78.107, 77.195,
(14,0): 78.005, 78.005, 78.005, 78.005, 78.005, 78.005, 76.991, 76.991, 76.991, 76.991
}
}
}
}
What if we wish to read three rows of three elements at a time (-c "3,3"), where each element is a 2 x 3 block (-k "2,3") and we wish to begin reading from the second row (-s "1,0")?
You can do that with the following command: 
h5dump -A 0 -d "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle"
-s "1,0" -S "2,3" -c "3,3" -k "2,3" -w 0 OMI-Aura.he5
In this case, the stride must be specified as 2 by 3 (or larger) to accommodate the reading of 2 by 3 blocks. If it is smaller, the command will fail with the error, 
h5dump error: wrong subset selection; blocks overlap.
The output of the above command is shown below: 
HDF5 "OMI-Aura.he5" {
DATASET "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/SolarZenithAngle" {
DATATYPE [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
DATASPACE SIMPLE { ( 720, 1440 ) / ( 720, 1440 ) }
SUBSET {
START ( 1, 0 );
STRIDE ( 2, 3 );
COUNT ( 3, 3 );
BLOCK ( 2, 3 );
DATA {
(1,0): 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071, 79.071,
(2,0): 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867, 78.867,
(3,0): 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632, 78.632,
(4,0): 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429, 78.429,
(5,0): 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225, 78.225,
(6,0): 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021, 78.021
}
}
}
}
#  Datatypes
##  h5dump
The following datatypes are discussed, using the output of `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` with HDF5 files from the [Examples by API](https://support.hdfgroup.org/documentation/hdf5/latest/_ex_a_p_i.html) page: 
  * [Array](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDtypes_array)
  * [Object Reference](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDtypes_objref)
  * [Region Reference](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDtypes_regref)
  * [String](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#subsubsecViewToolsViewDtypes_string)


###  Array
Users have been confused by the difference between an Array datatype ([H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)) and a dataset that (has a dataspace that) is an array.
Typically, these users want a dataset that has a simple datatype (like integer or float) that is an array, like the following dataset `/DS1`. It has a datatype of [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) (32-bit Little-Endian Integer) and is a 4 by 7 array: 
$ h5dump h5ex_d_rdwr.h5
HDF5 "h5ex_d_rdwr.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 4, 7 ) / ( 4, 7 ) }
DATA {
(0,0): 0, -1, -2, -3, -4, -5, -6,
(1,0): 0, 0, 0, 0, 0, 0, 0,
(2,0): 0, 1, 2, 3, 4, 5, 6,
(3,0): 0, 2, 4, 6, 8, 10, 12
}
}
}
} 
Contrast that with the following dataset that has both an Array datatype and is an array: 
$ h5dump h5ex_t_array.h5
HDF5 "h5ex_t_array.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) { [3][5] [H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565) }
DATASPACE SIMPLE { ( 4 ) / ( 4 ) }
DATA {
(0): [ 0, 0, 0, 0, 0,
0, -1, -2, -3, -4,
0, -2, -4, -6, -8 ],
(1): [ 0, 1, 2, 3, 4,
1, 1, 1, 1, 1,
2, 1, 0, -1, -2 ],
(2): [ 0, 2, 4, 6, 8,
2, 3, 4, 5, 6,
4, 4, 4, 4, 4 ],
(3): [ 0, 3, 6, 9, 12,
3, 5, 7, 9, 11,
6, 7, 8, 9, 10 ]
}
}
}
}
[H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)
@ H5T_ARRAY
**Definition** H5Tpublic.h:42
[H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565)
#define H5T_STD_I64LE
**Definition** H5Tpublic.h:481
In this file, dataset `/DS1` has a datatype of 
[H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) { [3][5] [H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565) }
and it also has a dataspace of 
SIMPLE { ( 4 ) / ( 4 ) }
In other words, it is an array of four elements, in which each element is a 3 by 5 array of [H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565).
This dataset is much more complex. Also note that subsetting cannot be done on Array datatypes.
See [Datatype Basics](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_datatypes.html) section for more information on the Array datatype.
###  New References
References were reworked in HDF5 1.12.0. The new reference datatype is [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc). The old reference datatypes are deprecated. see [HDF5 References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#sec_reference).
###  Object Reference
An Object Reference is a reference to an entire object (attribute, dataset, group, or named datatype). A dataset with an Object Reference datatype consists of one or more Object References. An Object Reference dataset can be used as an index to an HDF5 file.
The `/DS1` dataset in the following file (`h5ex_t_objref.h5`) is an Object Reference dataset. It contains two references, one to group `/G1` and the other to dataset `/DS2`: 
$ h5dump h5ex_t_objref.h5
HDF5 "h5ex_t_objref.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) { [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) }
DATASPACE SIMPLE { ( 2 ) / ( 2 ) }
DATA {
GROUP "h5ex_t_objref.h5/G1"
DATASET "h5ex_t_objref.h5/DS2"
DATA {
}
}
}
DATASET "DS2" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE NULL
DATA {
}
}
GROUP "G1" {
}
}
}
[H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd)
@ H5T_REFERENCE
**Definition** H5Tpublic.h:39
[H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc)
#define H5T_STD_REF
**Definition** H5Tpublic.h:576
###  Region Reference
A Region Reference is a reference to a selection within a dataset. A selection can be either individual elements or a hyperslab. In `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` you will see the name of the dataset along with the elements or slab that is selected. A dataset with a Region Reference datatype consists of one or more Region References.
An example of a Region Reference dataset (`h5ex_t_regref.h5`) can be found on the [Examples by API](https://support.hdfgroup.org/documentation/hdf5/latest/_ex_a_p_i.html) page, under Datatypes. If you examine this dataset with `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` you will see that `/DS1` is a Region Reference dataset as indicated by its datatype, highlighted in bold below: 
$ h5dump h5ex_t_regref.h5
HDF5 "h5ex_t_regref.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) { [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) }
DATASPACE SIMPLE { ( 2 ) / ( 2 ) }
DATA {
DATASET "h5ex_t_regref.h5/DS2"{
REGION_TYPE POINT (0,1), (2,11), (1,0), (2,4)
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
}
DATASET "h5ex_t_regref.h5/DS2" {
REGION_TYPE BLOCK (0,0)-(0,2), (0,11)-(0,13), (2,0)-(2,2),
(2,11)-(2,13)
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
}
}
}
DATASET "DS2" {
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
DATA {
(0,0): 84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119,
(0,14): 110, 0,
(1,0): 102, 111, 120, 32, 106, 117, 109, 112, 115, 32, 111, 118, 101,
(1,13): 114, 32, 0,
(2,0): 116, 104, 101, 32, 53, 32, 108, 97, 122, 121, 32, 100, 111, 103,
(2,14): 115, 0
}
}
}
}
[H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
#define H5T_STD_I8LE
**Definition** H5Tpublic.h:451
It contains two Region References: 
  * A selection of four individual elements in dataset `/DS2 : (0,1), (2,11), (1,0), (2,4)` See the [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d "Selects array elements to be included in the selection for a dataspace.") API in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) for information on selecting individual elements. 
  * A selection of these blocks in dataset `/DS2 : (0,0)-(0,2), (0,11)-(0,13), (2,0)-(2,2), (2,11)-(2,13)` See the [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.") API in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) for how to do hyperslab selection.


If you look at the code that creates the dataset (`h5ex_t_regref.c`) you will see that the first reference is created with these calls: 
status = [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d) (space, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), 4, coords[0]);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file, DATASET2, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &wdata[0]);
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550)
@ H5S_SELECT_SET
**Definition** H5Spublic.h:100
[H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)
herr_t H5Rcreate_region(hid_t loc_id, const char *name, hid_t space_id, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates a region reference.
[H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d)
herr_t H5Sselect_elements(hid_t space_id, H5S_seloper_t op, size_t num_elem, const hsize_t *coord)
Selects array elements to be included in the selection for a dataspace.
where the buffer containing the coordinates to select is: 
coords[4][2] = { {0, 1},
{2, 11},
{1, 0},
{2, 4} },
The second reference is created by calling, 
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d) (space, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), start, stride, count, block);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file, DATASET2, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &wdata[1]);
[H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)
herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op, const hsize_t start[], const hsize_t stride[], const hsize_t count[], const hsize_t block[])
Selects a hyperslab region to add to the current selected region.
where start, stride, count, and block have these values: 
start[2] = {0, 0},
stride[2] = {2, 11},
count[2] = {2, 2},
block[2] = {1, 3};
These start, stride, count, and block values will select the elements shown in bold in the dataset: 
84 104 101 32 113 117 105 99 107 32 98 114 111 119 110 0
102 111 120 32 106 117 109 112 115 32 111 118 101 114 32 0
116 104 101 32 53 32 108 97 122 121 32 100 111 103 115 0
If you use `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` to select a subset of dataset `/DS2` with these start, stride, count, and block values, you will see that the same elements are selected: 
$ h5dump -d "/DS2" -s "0,0" -S "2,11" -c "2,2" -k "1,3" h5ex_t_regref.h5
HDF5 "h5ex_t_regref.h5" {
DATASET "/DS2" {
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
SUBSET {
START ( 0, 0 );
STRIDE ( 2, 11 );
COUNT ( 2, 2 );
BLOCK ( 1, 3 );
DATA {
(0,0): 84, 104, 101, 114, 111, 119,
(2,0): 116, 104, 101, 100, 111, 103
}
}
}
}
NOTE that you must release the references created in the code with the [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095 "Closes a reference.") API. 
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[0]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[1]);
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)
herr_t H5Rdestroy(H5R_ref_t *ref_ptr)
Closes a reference.
For more information on selections, see the tutorial topic on [Reading From or Writing To a Subset of a Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_dset_sub_r_w.html). Also see the [Dataset Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewSub) tutorial topic on using `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` to view a subset.
###  Deprecated Object Reference
An Object Reference is a reference to an entire object (dataset, group, or named datatype). A dataset with an Object Reference datatype consists of one or more Object References. An Object Reference dataset can be used as an index to an HDF5 file.
The `/DS1` dataset in the following file (`h5ex_t_objref.h5`) is an Object Reference dataset. It contains two references, one to group `/G1` and the other to dataset `/DS2`: 
$ h5dump h5ex_t_objref.h5
HDF5 "h5ex_t_objref.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) { H5T_STD_REF_OBJECT }
DATASPACE SIMPLE { ( 2 ) / ( 2 ) }
DATA {
(0): GROUP 1400 /G1 , DATASET 800 /DS2
}
}
DATASET "DS2" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE NULL
DATA {
}
}
GROUP "G1" {
}
}
}
###  Deprecated Region Reference
A Region Reference is a reference to a selection within a dataset. A selection can be either individual elements or a hyperslab. In `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` you will see the name of the dataset along with the elements or slab that is selected. A dataset with a Region Reference datatype consists of one or more Region References.
An example of a Region Reference dataset (`h5ex_t_regref.h5`) can be found on the [Examples by API](https://support.hdfgroup.org/documentation/hdf5/latest/_ex_a_p_i.html) page, under Datatypes. If you examine this dataset with `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` you will see that `/DS1` is a Region Reference dataset as indicated by its datatype, highlighted in bold below: 
$ h5dump h5ex_t_regref.h5
HDF5 "h5ex_t_regref.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) { [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e) }
DATASPACE SIMPLE { ( 2 ) / ( 2 ) }
DATA {
DATASET /DS2 {(0,1), (2,11), (1,0), (2,4)},
DATASET /DS2 {(0,0)-(0,2), (0,11)-(0,13), (2,0)-(2,2), (2,11)-(2,13)}
}
}
DATASET "DS2" {
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
DATA {
(0,0): 84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119,
(0,14): 110, 0,
(1,0): 102, 111, 120, 32, 106, 117, 109, 112, 115, 32, 111, 118, 101,
(1,13): 114, 32, 0,
(2,0): 116, 104, 101, 32, 53, 32, 108, 97, 122, 121, 32, 100, 111, 103,
(2,14): 115, 0
}
}
}
}
[H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e)
#define H5T_STD_REF_DSETREG
**Definition** H5Tpublic.h:571
It contains two Region References: 
  * A selection of four individual elements in dataset `/DS2 : (0,1), (2,11), (1,0), (2,4)` See the [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d "Selects array elements to be included in the selection for a dataspace.") API in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) for information on selecting individual elements. 
  * A selection of these blocks in dataset `/DS2 : (0,0)-(0,2), (0,11)-(0,13), (2,0)-(2,2), (2,11)-(2,13)` See the [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.") API in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) for how to do hyperslab selection.


If you look at the code that creates the dataset (`h5ex_t_regref.c`) you will see that the first reference is created with these calls: 
status = [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d) (space, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), 4, coords[0]);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d) (&wdata[0], file, DATASET2, [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space);
[H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0)
#define H5R_DATASET_REGION
**Definition** H5Rpublic.h:625
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)
herr_t H5Rcreate(void *ref, hid_t loc_id, const char *name, H5R_type_t ref_type, hid_t space_id)
Creates a reference.
where the buffer containing the coordinates to select is: 
coords[4][2] = { {0, 1},
{2, 11},
{1, 0},
{2, 4} },
The second reference is created by calling, 
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d) (space, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), start, stride, count, block);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d) (&wdata[1], file, DATASET2, [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space);
where start, stride, count, and block have these values: 
start[2] = {0, 0},
stride[2] = {2, 11},
count[2] = {2, 2},
block[2] = {1, 3};
These start, stride, count, and block values will select the elements shown in bold in the dataset: 
84 104 101 32 113 117 105 99 107 32 98 114 111 119 110 0
102 111 120 32 106 117 109 112 115 32 111 118 101 114 32 0
116 104 101 32 53 32 108 97 122 121 32 100 111 103 115 0
If you use `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` to select a subset of dataset `/DS2` with these start, stride, count, and block values, you will see that the same elements are selected: 
$ h5dump -d "/DS2" -s "0,0" -S "2,11" -c "2,2" -k "1,3" h5ex_t_regref.h5
HDF5 "h5ex_t_regref.h5" {
DATASET "/DS2" {
DATATYPE [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
DATASPACE SIMPLE { ( 3, 16 ) / ( 3, 16 ) }
SUBSET {
START ( 0, 0 );
STRIDE ( 2, 11 );
COUNT ( 2, 2 );
BLOCK ( 1, 3 );
DATA {
(0,0): 84, 104, 101, 114, 111, 119,
(2,0): 116, 104, 101, 100, 111, 103
}
}
}
}
For more information on selections, see the tutorial topic on [Reading From or Writing To a Subset of a Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_dset_sub_r_w.html). Also see the [Dataset Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_view.html#secViewToolsViewSub) tutorial topic on using `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` to view a subset.
###  String
There are two types of string data, fixed length strings and variable length strings.
Below is the `h5dump[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump)` output for two files that have the same strings written to them. In one file, the strings are fixed in length, and in the other, the strings have different sizes (and are variable in size).
_Dataset of Fixed Length Strings_
HDF5 "h5ex_t_string.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) {
STRSIZE 7;
STRPAD [H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c);
CSET [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663);
CTYPE [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d);
}
DATASPACE SIMPLE { ( 4 ) / ( 4 ) }
DATA {
(0): "Parting", "is such", "sweet ", "sorrow."
}
}
}
}
[H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c)
@ H5T_STR_SPACEPAD
**Definition** H5Tpublic.h:123
_Dataset of Variable Length Strings_
HDF5 "h5ex_t_vlstring.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) {
STRSIZE [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc);
STRPAD [H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c);
CSET [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663);
CTYPE [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d);
}
DATASPACE SIMPLE { ( 4 ) / ( 4 ) }
DATA {
(0): "Parting", "is such", "sweet", "sorrow."
}
}
}
}
[H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc)
#define H5T_VARIABLE
**Definition** H5Tpublic.h:209
You might wonder which to use. Some comments to consider are included below. 
  * In general, a variable length string dataset is more complex than a fixed length string. If you don't specifically need a variable length type, then just use the fixed length string. 
  * A variable length dataset consists of pointers to heaps in different locations in the file. For this reason, a variable length dataset cannot be compressed. (Basically, the pointers get compressed and not the actual data!) If compression is needed, then do not use variable length types. 
  * If you need to create an array of of different length strings, you can either use fixed length strings along with compression, or use a variable length string.


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html) / [Command-line Tools](https://support.hdfgroup.org/documentation/hdf5/latest/_view_tools_command.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


