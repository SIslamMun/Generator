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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title2 "H2")
  * [◆ H5Mclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title3 "H2")
  * [◆ H5Mcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title4 "H2")
  * [◆ H5Mcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title5 "H2")
  * [◆ H5Mdelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title6 "H2")
  * [◆ H5Mexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title7 "H2")
  * [◆ H5Mget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title8 "H2")
  * [◆ H5Mget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title9 "H2")
  * [◆ H5Mget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title10 "H2")
  * [◆ H5Mget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title11 "H2")
  * [◆ H5Mget_key_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title12 "H2")
  * [◆ H5Mget_val_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title13 "H2")
  * [◆ H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title14 "H2")
  * [◆ H5Miterate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title15 "H2")
  * [◆ H5Mopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title16 "H2")
  * [◆ H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#title17 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#func-members)
VOL Mapping (H5M)
## Detailed Description
**The interface can only be used with the HDF5 VOL connectors that implement map objects.** The native HDF5 library does not support this feature.
While the HDF5 data model is a flexible way to store data, some applications require a more general way to index information. HDF5 effectively uses key-value stores internally for a variety of purposes, but it does not expose a generic key-value store to the API. The Map APIs provide this capability to the HDF5 applications in the form of HDF5 map objects. These Map objects contain application-defined key-value stores, to which key-value pairs can be added, and from which values can be retrieved by key.
HDF5 VOL connectors with support for map objects:
  * DAOS



Example:
    
// NOTE: This example requires a VOL connector with map support (e.g., DAOS)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, fapl_id, map_id, vls_type_id;
const char *names[2] = {"Alice", "Bob"};
uint64_t IDs[2] = {25385486, 34873275};
uint64_t val_out;
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) ret;
// Setup file access property list with VOL connector that supports maps
fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
// ... configure VOL connector in fapl_id (see connector documentation) ...
vls_type_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)(vls_type_id, [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc));
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl_id);
map_id = [H5Mcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09)(file_id, "map", vls_type_id, [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)(map_id, vls_type_id, &names[0], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &IDs[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)(map_id, vls_type_id, &names[1], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &IDs[1], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
ret = [H5Mget](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed)(map_id, vls_type_id, &names[0], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &val_out, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
if(ret < 0 || val_out != IDs[0]) {
fprintf(stderr, "Map retrieval failed\n");
// Handle error...
}
[H5Mclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44)(map_id);
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)(vls_type_id);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl_id);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file_id);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc)
#define H5T_VARIABLE
**Definition** H5Tpublic.h:209
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Mget](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed)
herr_t H5Mget(hid_t map_id, hid_t key_mem_type_id, const void *key, hid_t val_mem_type_id, void *value, hid_t dxpl_id)
Retrieves a key-value pair from a map object.
[H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)
herr_t H5Mput(hid_t map_id, hid_t key_mem_type_id, const void *key, hid_t val_mem_type_id, const void *value, hid_t dxpl_id)
Adds a key-value pair to a map object.
[H5Mcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09)
hid_t H5Mcreate(hid_t loc_id, const char *name, hid_t key_type_id, hid_t val_type_id, hid_t lcpl_id, hid_t mcpl_id, hid_t mapl_id)
Creates a map object.
[H5Mclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44)
herr_t H5Mclose(hid_t map_id)
Terminates access to a map object.
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)
herr_t H5Tset_size(hid_t type_id, size_t size)
Sets size for a datatype.
[H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)
hid_t H5Tcopy(hid_t type_id)
Copies an existing datatype.
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)
herr_t H5Tclose(hid_t type_id)
Releases a datatype.
[H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613)
#define H5T_NATIVE_UINT64
**Definition** H5Tpublic.h:1267
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d)
#define H5T_C_S1
**Definition** H5Tpublic.h:646
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id)  
| Terminates access to a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id)  
| Creates a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga978881a6ecbacc418232d318549307f2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id)  
| Creates a map object without linking it into a file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mdelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gae3cd428dc0746052779e7e372bfd2cad) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Deletes a key-value pair from a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga4754a39d7fd23715592df34ece9801f7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Checks if provided key exists in a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mget](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_mem_type_id, void *value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Retrieves a key-value pair from a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga48dd6941c1cac4f8f0f3beaf01c5d7de) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id)  
| Gets access property list for a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mget_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga1c6119224bdb84e4254f772bbee08ac0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *count, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Retrieves the number of key-value pairs in a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga5d05e2962702d80d27643abc05ee3da7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id)  
| Gets creation property list for a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mget_key_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga2001636fe6ce4c8aed13fa768c76a8fc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id)  
| Gets key datatype for a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mget_val_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga3c88ee008fa4ac1fa3b7824f182afc45) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id)  
| Gets value datatype for a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Miterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, [H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Iterates over all key-value pairs in a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Miterate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga9318fe7ee18d5630ad831868675c7ce9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *map_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, [H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Iterates over all key-value pairs in a map object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga426fbeac4c90849f2ac1855447fb8d43) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id)  
| Opens a map object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_mem_type_id, const void *value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id)  
| Adds a key-value pair to a map object.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44)H5Mclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _map_id_ | ) |   
---|---|---|---|---|---  
Terminates access to a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44 "Terminates access to a map object.") closes access to a map object specified by `map_id` and releases resources used by it.
It is illegal to subsequently use that same map identifier in calls to other map functions. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09)H5Mcreate()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mcreate  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ )  
Creates a map object.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Map object name   
[in] | key_type_id | Datatype identifier   
[in] | val_type_id | Datatype identifier   
[in] | lcpl_id | Link creation property list identifier   
[in] | mcpl_id | Map creation property list identifier   
[in] | mapl_id | Map access property list identifier  

Returns
    Returns a map object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09 "Creates a map object.") creates a new map object for storing key-value pairs. The in-file datatype for keys is defined by `key_type_id` and the in-file datatype for values is defined by `val_type_id`. `loc_id` specifies the location to create the map object and `name` specifies the name of the link to the map object relative to `loc_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga978881a6ecbacc418232d318549307f2)H5Mcreate_anon()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mcreate_anon  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ )  
Creates a map object without linking it into a file.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | key_type_id | Datatype identifier   
[in] | val_type_id | Datatype identifier   
[in] | mcpl_id | Map creation property list identifier   
[in] | mapl_id | Map access property list identifier  

Returns
    Returns a map object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). The resulting ID should be linked into the file with H5Olink or it will be deleted when closed.  
[H5Mcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga978881a6ecbacc418232d318549307f2 "Creates a map object without linking it into a file.") creates a new map object for storing key-value pairs. The in-file datatype for keys is defined by `key_type_id` and the in-file datatype for values is defined by `val_type_id`. `loc_id` specifies the file to create the map object, but no link to the object is created. Other options can be specified through the property lists `mcpl_id` and `mapl_id`.
The new map should be linked into the group hierarchy before being closed or it will be deleted. The map should be closed when the caller no longer requires it. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gae3cd428dc0746052779e7e372bfd2cad)H5Mdelete()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mdelete  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Deletes a key-value pair from a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[in] | key_mem_type_id | Datatype identifier   
[in] | key | Pointer to key buffer   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mdelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gae3cd428dc0746052779e7e372bfd2cad "Deletes a key-value pair from a map object.") deletes a key-value pair from the map object specified by `map_id`. `key_mem_type_id` specifies the datatype for the provided key buffer key, and if different from that used to create the map object, the key will be internally converted to the datatype for the map object.
Any further options can be specified through the property list `dxpl_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga4754a39d7fd23715592df34ece9801f7)H5Mexists()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mexists  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | bool * |  _exists_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Checks if provided key exists in a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[in] | key_mem_type_id | Datatype identifier   
[in] | key | Pointer to key buffer   
[out] | exists | Pointer to a buffer to return the existence status   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga4754a39d7fd23715592df34ece9801f7 "Checks if provided key exists in a map object.") checks if the provided key is stored in the map object specified by `map_id`. If `key_mem_type_id` is different from that used to create the map object the key will be internally converted to the datatype for the map object for the query.
Any further options can be specified through the property list `dxpl_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed)H5Mget()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mget  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_mem_type_id_ ,   
|  | void * |  _value_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Retrieves a key-value pair from a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[in] | key_mem_type_id | Datatype identifier   
[in] | key | Pointer to key buffer   
[in] | val_mem_type_id | Datatype identifier   
[out] | value | Pointer to value buffer   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed "Retrieves a key-value pair from a map object.") retrieves from a map object specified by `map_id`, the value associated with the provided key `key`. `key_mem_type_id` and `val_mem_type_id` specify the datatypes for the provided key and value buffers. If if the datatype specified by `key_mem_type_id` is different from that used to create the map object the key will be internally converted to the datatype for the map object for the query, and if the datatype specified by `val_mem_type_id` is different from that used to create the map object the returned value will be converted to have a datatype as specified by `val_mem_type_id` before the function returns.
Any further options can be specified through the property list `dxpl_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga48dd6941c1cac4f8f0f3beaf01c5d7de)H5Mget_access_plist()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mget_access_plist  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _map_id_ | ) |   
---|---|---|---|---|---  
Gets access property list for a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|--- 

Returns
    Returns a map access property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga48dd6941c1cac4f8f0f3beaf01c5d7de "Gets access property list for a map object.") returns an identifier for a copy of the access property list for a map object specified by `map_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga1c6119224bdb84e4254f772bbee08ac0)H5Mget_count()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mget_count  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _count_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Retrieves the number of key-value pairs in a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[out] | count | The number of key-value pairs stored in the map object   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga1c6119224bdb84e4254f772bbee08ac0 "Retrieves the number of key-value pairs in a map object.") retrieves the number of key-value pairs stored in a map specified by map_id. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga5d05e2962702d80d27643abc05ee3da7)H5Mget_create_plist()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mget_create_plist  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _map_id_ | ) |   
---|---|---|---|---|---  
Gets creation property list for a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|--- 

Returns
    Returns a map creation property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga5d05e2962702d80d27643abc05ee3da7 "Gets creation property list for a map object.") returns an identifier for a copy of the creation property list for a map object specified by `map_id`.
The creation property list identifier should be released with [H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") to prevent resource leaks. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga2001636fe6ce4c8aed13fa768c76a8fc)H5Mget_key_type()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mget_key_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _map_id_ | ) |   
---|---|---|---|---|---  
Gets key datatype for a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|--- 

Returns
    Returns a datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mget_key_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga2001636fe6ce4c8aed13fa768c76a8fc "Gets key datatype for a map object.") retrieves key datatype as stored in the file for a map object specified by `map_id` and returns identifier for the datatype. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga3c88ee008fa4ac1fa3b7824f182afc45)H5Mget_val_type()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mget_val_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _map_id_ | ) |   
---|---|---|---|---|---  
Gets value datatype for a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|--- 

Returns
    Returns a datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mget_val_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga3c88ee008fa4ac1fa3b7824f182afc45 "Gets value datatype for a map object.") retrieves value datatype as stored in the file for a map object specified by `map_id` and returns identifier for the datatype . 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2)H5Miterate()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Miterate  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _idx_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | [H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe) |  _op_ ,   
|  | void * |  _op_data_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Iterates over all key-value pairs in a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[in,out] | idx | iteration index   
[in] | key_mem_type_id | Datatype identifier   
[in] | op | User-defined iterator function   
[in,out] | op_data | User-defined callback function context   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") iterates over all key-value pairs stored in the map object specified by `map_id`, making the callback specified by `op` for each. The `idx` parameter is an in/out parameter that may be used to restart a previously interrupted iteration. At the start of iteration `idx` should be set to 0, and to restart iteration at the same location on a subsequent call to [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object."), `idx` should be the same value as returned by the previous call. Iterate callback is defined as: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, const void *key, void *op_data);
The `key` parameter is the buffer for the key for this iteration, converted to the datatype specified by `key_mem_type_id`. The `op_data` parameter is a simple pass through of the value passed to [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object."), which can be used to store application-defined data for iteration. A negative return value from this function will cause [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") to issue an error, while a positive return value will cause [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") to stop iterating and return this value without issuing an error. A return value of zero allows iteration to continue.
Any further options can be specified through the property list `dxpl_id`. 

Warning
    Adding or removing key-value pairs to the map during iteration will lead to undefined behavior. 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga9318fe7ee18d5630ad831868675c7ce9)H5Miterate_by_name()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Miterate_by_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _map_name_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _idx_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | [H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe) |  _op_ ,   
|  | void * |  _op_data_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Iterates over all key-value pairs in a map object.  

Parameters
     [in] | loc_id | Location identifier   
---|---|---  
[in] | map_name | Map object name relative to the location specified by `loc_id`  
[in,out] | idx | Iteration index   
[in] | key_mem_type_id | Datatype identifier   
[in] | op | User-defined iterator function   
[in,out] | op_data | User-defined callback function context   
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | lapl_id | Link access property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Miterate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga9318fe7ee18d5630ad831868675c7ce9 "Iterates over all key-value pairs in a map object.") iterates over all key-value pairs stored in the map object specified by `map_id`, making the callback specified by `op` for each. The `idx` parameter is an in/out parameter that may be used to restart a previously interrupted iteration. At the start of iteration `idx` should be set to 0, and to restart iteration at the same location on a subsequent call to [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object."), `idx` should be the same value as returned by the previous call. Iterate callback is defined as: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5M_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_mpublic_8h.html#a689d878943fd9f1844809def5cfa60fe))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, const void *key, void *op_data);
The`key` parameter is the buffer for the key for this iteration, converted to the datatype specified by `key_mem_type_id`. The `op_data` parameter is a simple pass through of the value passed to [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object."), which can be used to store application-defined data for iteration. A negative return value from this function will cause [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") to issue an error, while a positive return value will cause [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") to stop iterating and return this value without issuing an error. A return value of zero allows iteration to continue.
Any further options can be specified through the property list `dxpl_id`. 

Warning
    Adding or removing key-value pairs to the map during iteration will lead to undefined behavior. 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga426fbeac4c90849f2ac1855447fb8d43)H5Mopen()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mopen  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ )  
Opens a map object.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Map object name relative to `loc_id`  
[in] | mapl_id | Map access property list identifier  

Returns
    Returns a map object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Mopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga426fbeac4c90849f2ac1855447fb8d43 "Opens a map object.") finds a map object specified by `name` under the location specified by `loc_id`. The map object should be close with [H5Mclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44 "Terminates access to a map object.") when the application is not longer interested in accessing it. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)H5Mput()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mput  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_mem_type_id_ ,   
|  | const void * |  _value_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ )  
Adds a key-value pair to a map object.  

Parameters
     [in] | map_id | Map identifier   
---|---|---  
[in] | key_mem_type_id | Datatype identifier   
[in] | key | Pointer to key buffer   
[in] | val_mem_type_id | Datatype identifier   
[in] | value | Pointer to value buffer   
[in] | dxpl_id | Dataset transfer property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f "Adds a key-value pair to a map object.") adds a key-value pair to a map object specified by `map_id`, or updates the value for the specified key if one was set previously.
`key_mem_type_id` and `val_mem_type_id` specify the datatypes for the provided key and value buffers, and if different from those used to create the map object, the key and value will be internally converted to the datatypes for the map object.
Any further options can be specified through the property list `dxpl_id`. 

Since
    1.12.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


