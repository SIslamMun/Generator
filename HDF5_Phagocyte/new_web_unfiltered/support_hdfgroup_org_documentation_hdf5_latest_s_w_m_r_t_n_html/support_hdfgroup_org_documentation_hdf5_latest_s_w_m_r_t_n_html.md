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
  * [ Introduction to SWMR](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title0 "H1")
  * [ Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title1 "H2")
  * [ Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title2 "H2")
  * [ Limitations and Scope](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title3 "H2")
  * [ Tools for Working with SWMR](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title4 "H2")
  * [ Programming Example](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Introduction to Single-Writer/Multiple-Reader (SWMR)
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction to SWMR
The Single-Writer / Multiple-Reader (SWMR) feature enables multiple processes to read an HDF5 file while it is being written to (by a single process) without using locks or requiring communication between processes. ![tutr-swmr1.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutr-swmr1.png)
All communication between processes must be performed via the HDF5 file. The HDF5 file under SWMR access must reside on a system that complies with POSIX write() semantics.
The basic engineering challenge for this to work was to ensure that the readers of an HDF5 file always see a coherent (though possibly not up to date) HDF5 file.
The issue is that when writing data there is information in the metadata cache in addition to the physical file on disk: ![tutr-swmr2.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutr-swmr2.png)
However, the readers can only see the state contained in the physical file: ![tutr-swmr3.png](https://support.hdfgroup.org/documentation/hdf5/latest/tutr-swmr3.png)
The SWMR solution implements dependencies on when the metadata can be flushed to the file. This ensures that metadata cache flush operations occur in the proper order, so that there will never be internal file pointers in the physical file that point to invalid (unflushed) file addresses.
A beneficial side effect of using SWMR access is better fault tolerance. It is more difficult to corrupt a file when using SWMR.
##  Documentation
###  User Guide
[SWMR User Guide](https://support.hdfgroup.org/releases/hdf5/documentation/features/SWMR/HDF5_SWMR_Users_Guide.pdf)
###  HDF5 Library APIs
  * [H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe "Retrieves free-space section information for a file.") — Enables SWMR writing mode for a file 
  * [H5DOappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c "Appends data to a dataset along a specified dimension.") — Appends data to a dataset along a specified dimension 
  * [H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19 "Sets a callback function to invoke when an object flush occurs in the file.") — Sets a callback function to invoke when an object flush occurs in the file 
  * [H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e "Retrieves the object flush property values from the file access property list.") — Retrieves the object flush property values from the file access property list 
  * [H5Odisable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga0908be309da1fb4f771c1e264fac22ae "Prevents metadata entries for an HDF5 object from being flushed from the metadata cache to storage.") — Prevents metadata entries for an HDF5 object from being flushed from the metadata cache to storage 
  * [H5Oenable_mdc_flushes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga21014920bdabf6973e233796d7174156 "Enables flushing of dirty metadata entries from a file's metadata cache.") — Enables flushing of dirty metadata entries from a file's metadata cache 
  * [H5Oare_mdc_flushes_disabled](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gac3b3b768a1c50be46e63cf5a631b30e3 "Retrieves comment for specified object.") — Determines if an HDF5 object has had flushes of metadata entries disabled 


###  Tools
  * [h5watch](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__w_h__u_g.html#sec_cltools_h5watch) — Outputs new records appended to a dataset as the dataset grows 
  * [h5format_convert](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__f_c__u_g.html#sec_cltools_h5format_convert) — Converts the layout format version and chunked indexing types of datasets created with HDF5-1.10 so that applications built with HDF5-1.8 can access them 
  * [h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#sec_cltools_h5clear) — Clears superblock status_flags field, removes metadata cache image, prints EOA and EOF, or sets EOA of a file


###  Design Documents
##  Programming Model
Please be aware that the SWMR feature requires that an HDF5 file be created with the latest file format. See [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") for more information.
To use SWMR follow the the general programming model for creating and accessing HDF5 files and objects along with the steps described below.
###  SWMR Writer
The SWMR writer either opens an existing file and objects or creates them as follows.
Open an existing file: Call [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") using the [H5F_ACC_SWMR_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a6217b039eabc4e484cee9813cbd0d020) flag. Begin writing datasets. Periodically flush data.
Create a new file: Call [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") using the latest file format. Create groups, datasets and attributes, and then close the attributes. Call [H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe "Retrieves free-space section information for a file.") to start SWMR access to the file. Periodically flush data.
#### Example Code:
Create the file using the latest file format property: 
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
status = [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910) (fapl, [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0));
fid = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) (filename, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl);
// Create objects (files, datasets, ...).
// Close any attributes and named datatype objects.
// Groups and datasets may remain open before starting SWMR access to them.
// Start SWMR access to the file:
status = [H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe) (fid);
// Reopen the datasets and then start writing, periodically flushing data:
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dset_id, ...);
status = [H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7) (dset_id);
[H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)
@ H5F_LIBVER_LATEST
**Definition** H5Fpublic.h:187
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)
herr_t H5Pset_libver_bounds(hid_t plist_id, H5F_libver_t low, H5F_libver_t high)
Controls the range of library release versions used when creating objects in a file.
[H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7)
herr_t H5Dflush(hid_t dset_id)
Flushes all buffers associated with a dataset to disk.
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
[H5Fstart_swmr_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_w_m_r.html#ga159be34fbe7e4a959589310ef0196dfe)
herr_t H5Fstart_swmr_write(hid_t file_id)
Retrieves free-space section information for a file.
###  SWMR Reader
The SWMR reader must continually poll for new data:
Call [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") using the [H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402) flag. Poll, checking the size of the dataset to see if there is new data available for reading. Read new data, if any.
#### Example Code:
// Open the file using the SWMR read flag:
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) (filename, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) | [H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Open the dataset and then repeatedly poll the dataset, by getting the dimensions, reading new data, and refreshing:
dset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (...);
space_id = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a) (...);
while (...) {
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) (dset_id, ...);
status = [H5Drefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da) (dset_id);
space_id = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a) (...);
}
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394)
#define H5F_ACC_RDONLY
**Definition** H5Fpublic.h:28
[H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402)
#define H5F_ACC_SWMR_READ
**Definition** H5Fpublic.h:49
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Drefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da)
herr_t H5Drefresh(hid_t dset_id)
Refreshes all buffers associated with a dataset.
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
[H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)
hid_t H5Dget_space(hid_t dset_id)
Returns an identifier for a copy of the dataspace for a dataset.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
##  Limitations and Scope
An HDF5 file under SWMR access must reside on a system that complies with POSIX write() semantics. It is also limited in scope as follows.
The writer process is only allowed to modify raw data of existing datasets by; Appending data along any unlimited dimension. Modifying existing data The following operations are not allowed (and the corresponding HDF5 files will fail) 
  * The writer cannot add new objects to the file. 
  * The writer cannot delete objects in the file. 
  * The writer cannot modify or append data with variable length, string or region reference datatypes. 
  * File space recycling is not allowed. As a result the size of a file modified by a SWMR writer may be larger than a file modified by a non-SWMR writer.


##  Tools for Working with SWMR
Two new tools, [h5watch](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__w_h__u_g.html#sec_cltools_h5watch) and [h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#sec_cltools_h5clear), are available for use with SWMR. The other HDF5 utilities have also been modified to recognize SWMR 
  * The [h5watch](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__w_h__u_g.html#sec_cltools_h5watch) tool allows a user to monitor the growth of a dataset. 
  * The [h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#sec_cltools_h5clear) tool clears the status flags in the superblock of an HDF5 file. 
  * The rest of the HDF5 tools will exit gracefully but not work with SWMR otherwise.


##  Programming Example
A good example of using SWMR is included with the HDF5 tests in the source code. You can run it while reading the file it creates. If you then interrupt the application and reader and look at the resulting file, you will see that the file is still valid. Follow these steps: 
  * Download the HDF5 source code to a local directory on a filesystem (that complies with POSIX write() semantics). Build the software. No special configuration options are needed to use SWMR. 
  * Invoke two command terminal windows. In one window go into the bin directory of the built binaries. In the other window go into the test directory of the HDF5-1.10 source code that was just built. 
  * In the window in the test directory compile and run use_append_chunk.c. The example writes a three dimensional dataset by planes (with chunks of size 1 x 256 x 256). 
  * In the other window (in the bin directory) run [h5watch](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__w_h__u_g.html#sec_cltools_h5watch) on the file created by use_append_chunk.c (use_append_chunk.h5). It should be run while use_append_chunk is executing and you will see valid data displayed with h5watch. 
  * Interrupt use_append_chunk while it is running, and stop [h5watch](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__w_h__u_g.html#sec_cltools_h5watch). 
  * Use [h5clear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_r__u_g.html#sec_cltools_h5clear) to clear the status flags in the superblock of the HDF5 file (use_append_chunk.h5). 
  * View the file with [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump). You will see that it is a valid file even though the application did not close properly. It will contain data up to the point that it was interrupted.


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


