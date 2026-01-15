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
  * [ Introduction to VDS](https://support.hdfgroup.org/documentation/hdf5/latest/_v_d_s_t_n.html#title0 "H1")
  * [ Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_v_d_s_t_n.html#title1 "H2")
  * [ Programming Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_v_d_s_t_n.html#title2 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Introduction to the Virtual Dataset - VDS
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction to VDS
The HDF5 Virtual Dataset (VDS) feature enables users to access data in a collection of HDF5 files as a single HDF5 dataset and to use the HDF5 APIs to work with that dataset.
For example, your data may be collected into four files: ![tutrvds-multimgs.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutrvds-multimgs.png)
You can map the datasets in the four files into a single VDS that can be accessed just like any other dataset: ![tutrvds-snglimg.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutrvds-snglimg.png)
The mapping between a VDS and the HDF5 source datasets is persistent and transparent to an application. If a source file is missing the fill value will be displayed.
See the Virtual (VDS) Documentation for complete details regarding the VDS feature.
The VDS feature was implemented using hyperslab selection ([H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")). See the tutorial on Reading From or Writing to a Subset of a Dataset for more information on selecting hyperslabs.
##  Programming Model
To create a Virtual Dataset you simply follow the HDF5 programming model and add a few additional API calls to map the source code datasets to the VDS.
Following are the steps for creating a Virtual Dataset: 
  * Create the source datasets that will comprise the VDS 
  * Create the VDS: ‐ Define a datatype and dataspace (can be unlimited) 
  * Define the dataset creation property list (including fill value) 
  * (Repeat for each source dataset) Map elements from the source dataset to elements of the VDS 
  * Select elements in the source dataset (source selection) 
  * Select elements in the virtual dataset (destination selection) 
  * Map destination selections to source selections (see Functions for Working with a VDS) 
  * Call H5Dcreate using the properties defined above 
  * Access the VDS as a regular HDF5 dataset 
  * Close the VDS when finished


#### Functions for Working with a VDS
The [H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets.") API sets the mapping between virtual and source datasets. This is a dataset creation property list. Using this API will change the layout of the dataset to [H5D_VIRTUAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06ea5c213e4ca5ea394669873ce66f558ad4). As with specifying any dataset creation property list, an instance of the property list is created, modified, passed into the dataset creation call and then closed: 
dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
src_space = H5screate_simple ...
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d) (space, ...
status = [H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5) (dcpl, space, SRC_FILE[i], SRC_DATASET[i], src_space);
dset = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df) (file, DATASET, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (dcpl);
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5)
herr_t H5Pset_virtual(hid_t dcpl_id, hid_t vspace_id, const char *src_file_name, const char *src_dset_name, hid_t src_space_id)
Sets the mapping between virtual and source datasets.
[H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
Creates a new dataset and links it into the file.
[H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)
herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op, const hsize_t start[], const hsize_t stride[], const hsize_t count[], const hsize_t block[])
Selects a hyperslab region to add to the current selected region.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
There are several other APIs introduced with Virtual Datasets, including query functions. For details see the complete list of HDF5 library APIs that support Virtual Datasets.
#### Limitations
This feature was introduced in HDF5-1.10.
The number of source datasets is unlimited. However, there is a limit on the size of each source dataset.
##  Programming Examples
_Example 1_ This example creates three HDF5 files, each with a one-dimensional dataset of 6 elements. The datasets in these files are the source datasets that are then used to create a 4 x 6 Virtual Dataset with a fill value of -1. The first three rows of the VDS are mapped to the data from the three source datasets as shown below: ![tutrvds-ex.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutrvds-ex.png)
In this example the three source datasets are mapped to the VDS with this code: 
>
src_space = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae) (RANK1, dims, NULL);
for (i = 0; i &lt; 3; i++) {
start[0] = ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c))i;
// Select i-th row in the virtual dataset; selection in the source datasets is the same.
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d) (space, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), start, NULL, count, block);
status = [H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5) (dcpl, space, SRC_FILE[i], SRC_DATASET[i], src_space);
}
endcode>
After the VDS is created and closed, it is reopened. The property list is then queried to determine the
layout of the dataset and its mappings, and the data in the VDS is read and printed.
This example is in the HDF5 source code and can be obtained from here:
<h4>C Example</h4>
For details on compiling an HDF5 application: [ Compiling HDF5 Applications ]
<h4>Example 2</h4>
This example shows how to use a C-style printf statement for specifying multiple source datasets as one virtual
dataset. Only one mapping is required. In other words only one #[H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5) call is needed to map multiple datasets.
It creates a 2-dimensional unlimited VDS. Then it re-opens the file, makes queries, and reads the virtual dataset.
The source datasets are specified as A-0, A-1, A-2, and A-3. These are mapped to the virtual dataset with one call:
\code
status = [H5Pset_virtual](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5) (dcpl, vspace, SRCFILE, "A-%b", src_space);
[H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550)
@ H5S_SELECT_SET
**Definition** H5Spublic.h:100
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
The b indicates that the block count of the selection in the dimension should be used.
#### C Example
For details on compiling an HDF5 application: [ Compiling HDF5 Applications ]
Using [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) with a VDS The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) utility can be used to view a VDS. The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) output for a VDS looks exactly like that for any other dataset. If [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) cannot find a source dataset then the fill value will be displayed.
You can determine that a dataset is a VDS by looking at its properties with 
h5dump -p
It will display each source dataset mapping, beginning with Mapping 0. Below is an excerpt of the output of 
h5dump -p
on the vds.h5 file created in Example 1.You can see that the entire source file a.h5 is mapped to the first row of the VDS dataset.
![tutrvds-map.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutrvds-map.png)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


