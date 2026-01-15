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
  * [ HDF5 Map Object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title1 "H2")
  * [ Map Object Life Cycle](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title2 "H2")
  * [ Map Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title3 "H2")
  * [ Key and Value Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title4 "H2")
  * [ Example Usage](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title5 "H2")
  * [ Important Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 VOL Data Mapping
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
**The HDF5 Data Mapping can only be used with the HDF5 VOL connectors that implement map objects.** The native HDF5 library does not support this feature.
#  HDF5 Map Object
##  Introduction
While the HDF5 data model provides a flexible way to store structured data, some applications require a more general mechanism for indexing information with arbitrary keys. HDF5 Map objects address this need by providing application-defined key-value stores where key-value pairs can be added, retrieved by key, and iterated over.
Map objects are available only through VOL connectors that implement the map interface. The native HDF5 file format does not support map objects. VOL connectors with map support include DAOS and other storage systems optimized for key-value operations. 

See also
     [VOL Mapping (H5M)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html) Reference Manual
##  Map Object Life Cycle
Map objects follow a well-defined life cycle from creation through cleanup:
  1. **Creation**
     * Create a new map with [H5Mcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09 "Creates a map object.") or [H5Mcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga978881a6ecbacc418232d318549307f2 "Creates a map object without linking it into a file."): 
       * [H5Mcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09 "Creates a map object.") creates a map and links it into the file hierarchy with a specified name 
       * [H5Mcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga978881a6ecbacc418232d318549307f2 "Creates a map object without linking it into a file.") creates an anonymous map that must be linked with [H5Olink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2c97dd58e64b67d16325fceb7e02113f "Creates a hard link to an object in an HDF5 file.") before closing 
     * Specify the key datatype and value datatype during creation - these define how keys and values are stored in the map 
     * Optionally provide creation and access property lists (MCPL and MAPL) to control map behavior 
     * The function returns a map identifier ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) used for all subsequent operations 
  2. **Opening**
     * Open an existing map with [H5Mopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga426fbeac4c90849f2ac1855447fb8d43 "Opens a map object.") by providing its location and name 
     * Optionally specify a map access property list (MAPL) to control access behavior 
     * Multiple opens of the same map are allowed and each returns a separate identifier 
  3. **Data Operations**
     * **Adding/Updating** : Use [H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f "Adds a key-value pair to a map object.") to add new key-value pairs or update existing values 
       * Specify memory datatypes for the key and value being written 
       * If the key already exists, its value is updated 
       * If the key is new, a new key-value pair is added 
     * **Retrieving** : Use [H5Mget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed "Retrieves a key-value pair from a map object.") to retrieve a value by its key 
       * Provide a buffer to receive the value 
       * Specify memory datatypes for key and value 
       * Type conversion is performed if memory and stored datatypes differ 
     * **Checking Existence** : Use [H5Mexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga4754a39d7fd23715592df34ece9801f7 "Checks if provided key exists in a map object.") to check if a key exists without retrieving its value 
     * **Deleting** : Use [H5Mdelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gae3cd428dc0746052779e7e372bfd2cad "Deletes a key-value pair from a map object.") to remove a key-value pair from the map 
     * **Iterating** : Use [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") or [H5Miterate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga9318fe7ee18d5630ad831868675c7ce9 "Iterates over all key-value pairs in a map object.") to process all key-value pairs 
       * Provide a callback function that is invoked for each key-value pair 
       * Iteration order is determined by the VOL connector implementation 
       * Can start iteration from a specific index position 
  4. **Query Operations**
     * [H5Mget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga1c6119224bdb84e4254f772bbee08ac0 "Retrieves the number of key-value pairs in a map object.") retrieves the number of key-value pairs stored in the map 
     * [H5Mget_key_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga2001636fe6ce4c8aed13fa768c76a8fc "Gets key datatype for a map object.") returns the datatype used for keys 
     * [H5Mget_val_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga3c88ee008fa4ac1fa3b7824f182afc45 "Gets value datatype for a map object.") returns the datatype used for values 
     * [H5Mget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga5d05e2962702d80d27643abc05ee3da7 "Gets creation property list for a map object.") retrieves the map creation property list (MCPL) 
     * [H5Mget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga48dd6941c1cac4f8f0f3beaf01c5d7de "Gets access property list for a map object.") retrieves the map access property list (MAPL) 
  5. **Closing**
     * Close the map with [H5Mclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44 "Terminates access to a map object.") when finished 
     * All map identifiers should be closed before closing the file 
     * Closing a map does not delete it from the file 
     * The library maintains reference counts and will not actually close the map until all identifiers are closed 


##  Map Property Lists
Map objects use two types of property lists that fit into the HDF5 property list class hierarchy (see [Property List Classes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#subsubsec_plist_class)):
  * **Map Creation Property List (MCPL) -[H5P_MAP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a50729bf4bfe1588f1ef3919e18e7d546)**
    * Controls properties that are set when a map is created 
    * These properties are permanent characteristics of the map 
    * Inherits from the Object Creation Property List (OCPL) class 
    * Can be retrieved later with [H5Mget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga5d05e2962702d80d27643abc05ee3da7 "Gets creation property list for a map object.")
    * Default MCPL: [H5P_MAP_CREATE_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aae5f6717cbedd91695b16d418cba6609) or [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
  * **Map Access Property List (MAPL) -[H5P_MAP_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a79320cf1e087a106edf5389499366ab3)**
    * Controls properties for accessing an existing map 
    * These properties affect how the map is opened and accessed 
    * Settings can vary between different opens of the same map 
    * Can be retrieved with [H5Mget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga48dd6941c1cac4f8f0f3beaf01c5d7de "Gets access property list for a map object.")
    * Default MAPL: [H5P_MAP_ACCESS_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#acdbe2a2c6e0695f8fc7e38fd727afc80) or [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)


These property list classes follow the same inheritance hierarchy as other HDF5 objects. The MCPL inherits properties relevant to object creation (like character encoding for names), while the MAPL controls access-specific settings.
##  Key and Value Datatypes
Map objects require two datatypes:
  * **Key Datatype** : Defines how keys are stored in the map 
    * Can be any valid HDF5 datatype (integer, float, string, compound, etc.) 
    * Variable-length strings ([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) with [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc) size) are commonly used for keys 
    * Fixed at map creation time 
  * **Value Datatype** : Defines how values are stored in the map 
    * Can be any valid HDF5 datatype 
    * All values in a map must have the same datatype 
    * Fixed at map creation time 


During [H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f "Adds a key-value pair to a map object.") and [H5Mget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed "Retrieves a key-value pair from a map object.") operations, memory datatypes can differ from the stored datatypes. The HDF5 library will perform type conversion as needed, similar to dataset I/O operations.
##  Example Usage
The following example demonstrates creating a map, adding key-value pairs, and retrieving values. Note that this example requires a VOL connector with map support (e.g., DAOS). See [HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html) for VOL connector configuration details.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, map_id, vls_type_id, fapl_id;
const char *names[2] = {"Alice", "Bob"};
uint64_t IDs[2] = {25385486, 34873275};
uint64_t val_out;
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) ret;
// Setup file access property list with VOL connector that supports maps
// (Configuration depends on specific VOL connector - see connector documentation)
fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
// ... configure VOL connector in fapl_id ...
// Create variable-length string datatype for keys
vls_type_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)(vls_type_id, [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc));
// Create file
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl_id);
// Create map with string keys and uint64 values
map_id = [H5Mcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09)(file_id, "map", vls_type_id, [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Add key-value pairs
[H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)(map_id, vls_type_id, &names[0], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &IDs[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Mput](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f)(map_id, vls_type_id, &names[1], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &IDs[1], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Retrieve a value by key
ret = [H5Mget](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed)(map_id, vls_type_id, &names[0], [H5T_NATIVE_UINT64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c9x.html#ga5eda31dcf868e6b9487d1486ec4a5613), &val_out, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
if(ret < 0 || val_out != IDs[0]) {
fprintf(stderr, "Failed to retrieve correct value from map\n");
goto error;
}
// Close map and other objects
[H5Mclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44)(map_id);
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)(vls_type_id);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl_id);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file_id);
return 0;
error:
[H5E_BEGIN_TRY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aec70809990a203fe2e6b044437f74551) {
[H5Mclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44)(map_id);
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)(vls_type_id);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl_id);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file_id);
} [H5E_END_TRY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#abed534bb6c70e3ac080b93e19343a3a5);
return -1;
[H5E_END_TRY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#abed534bb6c70e3ac080b93e19343a3a5)
#define H5E_END_TRY
**Definition** H5Epublic.h:103
[H5E_BEGIN_TRY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aec70809990a203fe2e6b044437f74551)
#define H5E_BEGIN_TRY
**Definition** H5Epublic.h:84
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
##  Important Notes
  * **VOL Connector Requirement** : Map functionality is only available through VOL connectors that implement the map interface. The native HDF5 file format does not support maps.
  * **Experimental API** : The H5M interface is experimental and subject to change in future releases. Application code using maps should be prepared for potential API modifications.
  * **Key Uniqueness** : Each key in a map must be unique. Calling [H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f "Adds a key-value pair to a map object.") with an existing key will update that key's value rather than creating a duplicate entry.
  * **Iteration Order** : The order in which key-value pairs are returned during iteration is determined by the VOL connector implementation and may not be predictable.
  * **Type Conversion** : As with datasets, the library performs datatype conversion between memory and storage representations. Ensure memory buffers are sized appropriately for the memory datatype specified.
  * **Asynchronous Operations** : Several map operations have asynchronous variants (e.g., [H5Mcreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8bb9c2b6c4e695156c529532e2984184), [H5Mopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga128bb64acfc9bf04191424c673da358d)) for use with the asynchronous I/O interface. 


Previous Chapter [The HDF5 Event Set Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_s__u_g.html#sec_async) - Next Chapter [HDF5 References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#sec_reference)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


