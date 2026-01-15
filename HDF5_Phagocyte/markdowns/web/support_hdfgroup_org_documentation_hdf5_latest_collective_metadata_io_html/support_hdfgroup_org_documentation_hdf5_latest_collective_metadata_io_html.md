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
  * [ Collective Metadata I/O Overview](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#title0 "H1")
  * [ Collective Metadata I/O User and Resource Documents](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#title1 "H1")
  * [ HDF5 Library APIs](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#title2 "H1")
  * [ New Collective Metadata I/O Functions](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#title3 "H2")
  * [ Additional API Reference](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Collective Metadata I/O
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
The purpose of this page is to briefly describe the new HDF5 Collective Metadata I/O feature and provide a gateway to available documentation. The page includes the following sections: 
  * [Collective Metadata I/O Overview](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#sec_collective_metadata_io_overview)
  * [Collective Metadata I/O User and Resource Documents](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#sec_collective_metadata_io_resource)
  * [HDF5 Library APIs](https://support.hdfgroup.org/documentation/hdf5/latest/collective_metadata_io.html#sec_collective_metadata_io_apis)


#  Collective Metadata I/O Overview
Calls for HDF5 metadata can result in many small reads and writes. On metadata reads, collective metadata I/O can improve performance by allowing the library to perform optimizations when reading the metadata, by having one rank read the data and broadcasting it to all other ranks.
Collective metadata I/O improves metadata write performance through the construction of an MPI derived datatype that is then written collectively in a single call.
#  Collective Metadata I/O User and Resource Documents
HDF5 Collective Metadata I/O User Document (This document is not yet available.)
Until an _HDF5 Collective Metadata I/O User Document_ becomes available, users may find the following resources helpful: 
  * [Collective Metadata Writes](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC-CollectiveMetadataWrites.pdf)
  * [Enabling Collective Metadata Reads](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC-CollectiveMetadataReads.pdf)


Taken together, these papers discuss the motivation, design, implementation, and API for HDF5’s Collective Metadata I/O feature.
#  HDF5 Library APIs
##  New Collective Metadata I/O Functions
API  | Description   
---|---  
[H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)") | Establishes I/O mode property setting, collective or independent, for metadata writes   
[H5Pget_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.") | Retrieves I/O mode property setting for metadata writes   
[H5Pset_all_coll_metadata_ops](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316 "Sets metadata I/O mode for read operations to be collective or independent \(default\)") | Establishes I/O mode, collective or independent, for metadata read operations   
[H5Pget_all_coll_metadata_ops](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gaa278583dd1a989fa35a915def6518126 "Retrieves metadata read mode setting.") | Retrieves I/O mode for metadata read operations   
##  Additional API Reference
###  Functions with No Access Property List Parameter that May Generate Metadata Reads
Currently there are several operations in HDF5 that can issue metadata reads from the metadata cache, but that take no property list. It is therefore not possible set a collective requirement individually for those operations. The only solution with the HDF5 1.10.0 release is to set the collective requirement globally on [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") or [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") for all metadata operations to be collective.
The following is a list of those functions in the HDF5 library. This list is integral to the discussion in the [H5Pset_all_coll_metadata_ops](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316 "Sets metadata I/O mode for read operations to be collective or independent \(default\)") entry: 
API   
---  
Attributes   
[H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c "Writes data to an attribute.")  
[H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c "Reads the value of an attribute.")  
[H5Arename](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga490dcd6db246c1fda7295badfce28203 "Renames an attribute.")  
[H5Aiterate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251 "Calls a user-defined function for each attribute on an object.")  
[H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba "Deletes an attribute from a specified location.")  
[H5Aexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga293b5be270d90cd5e47f782ca9aec80b "Determines whether an attribute with a given name exists on an object.")  
Datasets   
[H5Dget_space_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.")  
[H5Dget_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.")  
[H5Dset_extent](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.")  
H5Ddebug   
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.")  
[H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163 "Returns an identifier for a copy of the dataset creation property list for a dataset.")  
[H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.") (when dataset is a virtual dataset)   
Groups   
[H5Gget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga0b959a53cbffa48f5d68ce33b43b7ed8 "Gets a group creation property list identifier.")  
[H5Gget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad4be126ab7bbf2001435e8e70089f3d3 "Retrieves information about a group.")  
[H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.")  
Links   
[H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051)  
[H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69)  
References   
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d "Creates a reference.")  
[H5Rdereference2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7 "Opens the HDF5 object referenced.") (when reference is an object reference)   
[H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f "Sets up a dataspace and selection as specified by a region reference.")  
[H5Rget_obj_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807 "Retrieves the type of object that an object reference points to.")  
[H5Rget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga4c112c388f697324270fd085100dccaa "Retrieves a name for a referenced object.")  
Objects   
[H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")  
[H5Oopen_by_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga137f3823adab4daaaf8fe87b40453fa2 "Opens an object using its address within an HDF5 file.")  
[H5Oincr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2086bad6c3cd2a711c306a48c093ff55 "Increments an object reference count.")  
[H5Odecr_refcount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga60c20da5e244c28a653d4fa23d316b44 "Decrements an object reference count.")  
[H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5)  
[H5Oset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga8b5cf8e916204e29616516046121f631 "Sets comment for specified object.")  
[H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233)  
Files   
[H5Fis_hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga6055c2ea3438bd4aaf221eba66843225 "Determines whether a file is in the HDF5 format.")  
[H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.")  
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.")  
[H5Fget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file.")  
[H5Freopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3f213eb05c5419d63ba168c30036e47b "Returns a new identifier for a previously-opened HDF5 file.")  
[H5Fget_freespace](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3ef2673183567543346668a8f1eca2e9 "Returns the amount of free space in a file \(in bytes\)")  
[H5Fget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e "Retrieves global file information.")  
[H5Fget_free_sections](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gab9cbf1a45f9dcda34b43f985b7848434 "Retrieves free-space section information for a file.")  
[H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.")  
[H5Funmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga79e80cbf6d9782402b353b763162779c "Un-mounts an HDF5 file.")  
Identifiers   
[H5Iget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.")  
Datatypes   
[H5Tget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga6802c22c6e90216aa839a4a83909a54c "Returns a copy of a datatype's creation property list.")  
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0 "Releases a datatype.")  
Filters   
[H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8 "Unregisters a filter.")  
In addition, most deprecated functions fall into this category.
The HDF Group may address the above limitation in a future major release, but no decision has been made at this time. Such a change might, for example, include adding new versions of some or all the above functions with an extra property list parameter to allow an individual setting for the collective calling requirement.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


