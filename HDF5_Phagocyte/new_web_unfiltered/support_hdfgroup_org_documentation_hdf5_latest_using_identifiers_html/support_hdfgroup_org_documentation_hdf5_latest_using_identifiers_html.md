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
  * [ Using Identifiers](https://support.hdfgroup.org/documentation/hdf5/latest/_using_identifiers.html#title0 "H1")
  * [ Functions that Return Identifiers](https://support.hdfgroup.org/documentation/hdf5/latest/_using_identifiers.html#title1 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Using Identifiers
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
* * *
#  Using Identifiers
The purpose of this topic is to describe how identifiers behave and how they should be treated by application programs.
When an application program uses the HDF5 library to create or open an item, a unique identifier is returned. The items that return a unique identifier when they are created or opened include the following: 
  * dataset 
  * group 
  * datatype 
  * dataspace 
  * file 
  * attribute 
  * property list 
  * referenced object 
  * error stack 
  * error message


An application may open one of the items listed above more than once at the same time. For example, an application might open a group twice, receiving two identifiers. Information from one dataset in the group could be handled through one identifier, and the information from another dataset in the group could be handled by a different identifier.
An application program should track every identifier it receives as a result of creating or opening one of the items listed above. In order for an application to close properly, it must release every identifier it has opened. If an application opened a group twice for example, it would need to issue two [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.") commands, one for each identifier. Not releasing identifiers causes resource leaks. Until an identifier is released, the item associated with the identifier is still open.
The library considers a file open until all of the identifiers associated with the file and with the file’s various items have been released. The identifiers associated with these open items must be released separately. This means that an application can close a file and still work with one or more portions of the file. Suppose an application opened a file, a group within the file, and two datasets within the group. If the application closed the file with [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."), then the file would be considered closed to the application, but the group and two datasets would still be open.
There are several exceptions to the above file closing rule. One is when the [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") function is used instead of [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."). [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") causes a general shut down of the library: all data is written to disk, all identifiers are closed, and all memory used by the library is cleaned up. Another exception occurs on parallel processing systems. Suppose on a parallel system an application has opened a file, a group in the file, and two datasets in the group. If the application uses the [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") function to close the file, the call will fail with an error. The open group and datasets must be closed before the file can be closed. A third exception is when the file access property list includes the property [H5F_CLOSE_STRONG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fae6af53249bfe320745828497f28b6390). This property causes the closing of all of the file’s open items when the file is closed with [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."). For more information about [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory."), [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."), and [H5Pset_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree."), see the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
The reference manual entries for functions that return identifiers describe what might be returned as follows: **Returns:** Returns an identifier if successful; otherwise returns a negative value.
In other words, a successful operation will return a non-negative identifier which will never be 0 (zero) and will always be a positive value.
##  Functions that Return Identifiers
Some of the functions that return identifiers are listed below.
  * [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)
  * [H5Acreate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga004160c28e281455ec48aa7fe557ef8a "Creates an attribute attached to a specified object.")
  * [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44 "Gets an attribute's datatype.")
  * [H5Aopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga59863b205b6d93b2145f0fbca49656f7 "Opens an attribute for an object specified by object identifier and attribute name.")
  * [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d "Opens the nth attribute attached to an object.")
  * [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10 "Opens an attribute for an object by object name and attribute name.")
  * [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
  * [H5Dcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.")
  * [H5Dget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b "Returns the dataset access property list associated with a dataset.")
  * [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163 "Returns an identifier for a copy of the dataset creation property list for a dataset.")
  * [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.")
  * [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.")
  * [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
  * [H5Ecreate_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga99a705d98873dcdd1bb6f9d5eebc5afd "Adds a major or minor error message to an error class.")
  * [H5Ecreate_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga8bfca811dc01e97b4ab0736dd8775b39 "Creates a new, empty error stack.")
  * [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.")
  * [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.")
  * [H5Freopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3f213eb05c5419d63ba168c30036e47b "Returns a new identifier for a previously-opened HDF5 file.")
  * [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)
  * [H5Gcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gab52641f0736281faaaae4e3039bbb344 "Creates a new empty group without linking it into the file structure.")
  * [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)
  * [H5Oopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9f635f58c7ddf17f87c253bfbca08bc1 "Opens an object in an HDF5 file by location identifier and path name.")
  * [H5Oopen_by_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga137f3823adab4daaaf8fe87b40453fa2 "Opens an object using its address within an HDF5 file.")
  * [H5Oopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb66e5cbb3ca79890fc284a0b06762be "Opens the nth object in a group.")
  * [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.")
  * [H5Pget_virtual_srcspace](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping.")
  * [H5Pget_virtual_vspace](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping.")
  * [H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)
  * [H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f "Sets up a dataspace and selection as specified by a region reference.")
  * [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083 "Creates a new dataspace of a specified type.")
  * [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae "Creates a new simple dataspace and opens it for access.")
  * [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e "Copies an existing datatype.")
  * [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65 "Creates a new datatype.")
  * [H5Tdecode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac90f7722dacad861c0a22507d3adf0dd)
  * [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb "Returns the datatype of the specified member.")
  * [H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3 "Returns the base datatype from which a datatype is derived.")
  * [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209)


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) / [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


