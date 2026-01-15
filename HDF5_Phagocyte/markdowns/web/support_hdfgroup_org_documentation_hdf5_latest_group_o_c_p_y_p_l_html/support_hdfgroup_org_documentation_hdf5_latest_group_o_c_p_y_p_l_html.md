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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title2 "H2")
  * [◆ H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title3 "H2")
  * [◆ H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title4 "H2")
  * [◆ H5Pget_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title5 "H2")
  * [◆ H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title6 "H2")
  * [◆ H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title7 "H2")
  * [◆ H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#func-members)
Object Copy Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html)
## Detailed Description
Object copy property list functions (H5P) Function  | Purpose   
---|---  
[H5Padd_merge_committed_dtype_path](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...") | Adds a path to the list of paths that will be searched in the destination file for a matching committed datatype.   
[H5Pfree_merge_committed_dtype_paths](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.") | Clears the list of paths stored in the object copy property list.   
[H5Pset_copy_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")/[H5Pget_copy_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gad81b509481ba53a1ef1ba3c7083fc295 "Retrieves the properties to be used when an object is copied.") | Sets/gets the properties to be used when an object is copied.   
[H5Pset_mcdt_search_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")/[H5Pget_mcdt_search_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.") | Sets/gets the callback function that [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will invoke before searching for a matching committed datatype.   
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Padd_merge_committed_dtype_path](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, const char *path)  
| Adds a path to the list of paths that will be searched in the destination file for a matching committed datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pfree_merge_committed_dtype_paths](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Clears the list of paths stored in the object copy property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_copy_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gad81b509481ba53a1ef1ba3c7083fc295) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *copy_options)  
| Retrieves the properties to be used when an object is copied.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_mcdt_search_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb) *func, void **op_data)  
| Retrieves the callback function from the specified object copy property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_copy_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned copy_options)  
| Sets properties to be used when an object is copied.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_mcdt_search_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb) func, void *op_data)  
| Sets the callback function that [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will invoke before searching the entire destination file for a matching committed datatype.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f)H5Padd_merge_committed_dtype_path()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Padd_merge_committed_dtype_path  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | const char * |  _path_ )  
Adds a path to the list of paths that will be searched in the destination file for a matching committed datatype.  

Parameters
     [in] | plist_id | Object copy property list identifier   
---|---|---  
[in] | path | The path to be added 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...") adds a path, `path`, which points to a committed datatype, to the current list of suggested paths stored in the object copy property list `plist_id`. The search as described in the next paragraph is effective only if the [H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a11355b69cd246ae15e7d4c53cd210bd8) is enabled in the object copy property list via [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.").
When copying a committed datatype, a dataset with a committed datatype, or an object with an attribute of a committed datatype, the default behavior of [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") is to search for a matching committed datatype: 
  1. First search the list of suggested paths in the object copy property list. 
  2. Then, if no match has been found, search all the committed datatypes in the destination file. 


The default Step 2 in this search process can be changed by setting a callback function (see [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")).
Two datatypes are determined equal if their descriptions are identical, in a manner similar to [H5Tequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f "Determines whether two datatype identifiers refer to the same datatype."). If either committed datatype has one or more attributes, then all attributes must be present in both committed datatypes and they must be identical. Two attributes are considered identical if their datatype description, dataspace, and raw data values are the same. However, if an attribute uses a committed datatype, that committed datatype's attributes will not be compared.
If a match is found, [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will perform the following in the destination file: 
  * For a committed datatype, the library will create a hard link to the found datatype. 
  * For a dataset that uses a committed datatype, the library will modify the copied dataset to use the found committed datatype as its datatype. 
  * For an object with an attribute of a committed datatype, the library will modify the copied object's attribute to use the found committed datatype as its datatype.


If no match is found, [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will perform the following in the destination file: 
  * For a committed datatype, the library will copy it as it would any other object, creating a named committed datatype at the destination. That is, the library will create a committed datatype that is accessible in the file by a unique path. 
  * For a dataset that uses a committed datatype, the library will copy the datatype as an anonymous committed datatype and use that as the dataset's datatype. 
  * For an object with an attribute of a committed datatype, the library will copy the datatype as an anonymous committed datatype and use that as the attribute's datatype.


**Motivation:** [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...") provides a means to override the default behavior of [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") when [H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a11355b69cd246ae15e7d4c53cd210bd8) is set in an object copy property list. [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...") is the mechanism for suggesting search paths where [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will look for a matching committed datatype. This can be substantially faster than the default approach of searching the entire destination file for a match.
**Example** **Usage:** This example adds two paths to the object copy property list. [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will search the two suggested paths for a match before searching all the committed datatypes in the destination file.
```
    int main(void) {
    hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);

       H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
       H5Padd_merge_committed_dtype_path(ocpypl_id, "/group/committed_dtypeA");
       H5Padd_merge_committed_dtype_path(ocpypl_id, "/group2/committed_dset");
       H5Ocopy(...ocpypl_id...);
       ...
       ...
    }
    
```


Note
     [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...") will fail if the object copy property list is invalid. It will also fail if there is insufficient memory when duplicating `path`. 

See also
    
  * [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")
  * [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb)
  * [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...")
  * [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.")
  * [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.")
  * [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")
  * [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")
  * [Copying Committed Datatypes with H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/copying_committed.html)



Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf)H5Pfree_merge_committed_dtype_paths()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pfree_merge_committed_dtype_paths  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Clears the list of paths stored in the object copy property list.  

Parameters
     [in] | plist_id | Object copy property list identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.") clears the suggested paths stored in the object copy property list `plist_id`. These are the suggested paths previously set with [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...").
**Example** **Usage:** This example adds a suggested path to the object copy property list, does the copy, clears the list, and then adds a new suggested path to the list for another copy.
```
      int main(void) {
          hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);

          H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
          H5Padd_merge_committed_dtype_path(ocpypl_id, "/group/committed_dtypeA");
          H5Ocopy(...ocpypl_id...);
          ...
          ...
          H5Pfree_merge_committed_dtype_paths(ocpypl_id);
          H5Padd_merge_committed_dtype_path(ocpypl_id, "/group2/committed_dtypeB");
          H5Ocopy(...ocpypl_id...);
          ...
          ...
      }
      
```


Note
     [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.") will fail if the object copy property list is invalid. 

See also
    
  * [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")
  * [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb)
  * [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...")
  * [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.")
  * [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.")
  * [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")
  * [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")



Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gad81b509481ba53a1ef1ba3c7083fc295)H5Pget_copy_object()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_copy_object  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _copy_options_ )  
Retrieves the properties to be used when an object is copied.  

Parameters
     [in] | plist_id | Object copy property list identifier   
---|---|---  
[out] | copy_options | Copy option(s) set in the object copy property list 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gad81b509481ba53a1ef1ba3c7083fc295 "Retrieves the properties to be used when an object is copied.") retrieves the properties currently specified in the object copy property list `plist_id`, which will be invoked when a new copy is made of an existing object.
`copy_options` is a bit map indicating the flags, or properties, governing object copying that are set in the property list `plist_id`.
The available flags are described in [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied."). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396)H5Pget_mcdt_search_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_mcdt_search_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb) * |  _func_ ,   
|  | void ** |  _op_data_ )  
Retrieves the callback function from the specified object copy property list.  

Parameters
     [in] | plist_id | Object copy property list identifier   
---|---|---  
[out] | func | User-defined callback function   
[out] | op_data | User-defined data for the callback function 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.") retrieves the user-defined callback function and the user data that are set via [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...") in the object copy property list `plist_id`.
The callback function will be returned in the parameter `func` and the user data will be returned in the parameter `op_data`. 

Note
     [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.") will fail if the object copy property list is invalid. 

See also
    
  * [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")
  * [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb)
  * [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...")
  * [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.")
  * [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.")
  * [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")
  * [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")
  * [Copying Committed Datatypes with H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/copying_committed.html)



Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6)H5Pset_copy_object()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_copy_object  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _copy_options_ )  
Sets properties to be used when an object is copied.  

Parameters
     [in] | plist_id | Object copy property list identifier   
---|---|---  
[out] | copy_options | Copy option(s) to be set 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.") sets properties in the object copy property list `plist_id`. When an existing object is copied, that property list will determine how the new copy is created.
The following flags are available for use in an object copy property list:
[H5O_COPY_SHALLOW_HIERARCHY_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#af17242deb8f10196a580daf25daa17a0) | Copy only immediate members of a group  
_Default behavior, without flag:_ Recursively copy all objects in and below the group.   
---|---  
[H5O_COPY_EXPAND_SOFT_LINK_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a649171ae36665d655d9f13906462c736) | Expand soft links into new objects  
_Default behavior, without flag:_ Copy soft links as they are.   
[H5O_COPY_EXPAND_EXT_LINK_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a52b583ea3c1dd9cbc49a8c36a0383ca6) | Expand external link into new objects  
_Default behavior, without flag:_ Copy external links as they are.   
[H5O_COPY_EXPAND_REFERENCE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a8e951bb94b78613a9f4d90b53287f024) | Copy objects that are pointed to by references and update reference values in destination file  
_Default behavior, without flag:_ Set reference values in destination file to zero (0)   
[H5O_COPY_WITHOUT_ATTR_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#abc54bab3b27a232d430a3e836506de4c) | Copy object without copying attributes  
_Default behavior, without flag:_ Copy object with all its attributes   
[H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a11355b69cd246ae15e7d4c53cd210bd8) |  Use a matching committed datatype in the destination file when copying a committed datatype, a dataset with a committed datatype, or an object with an attribute of committed datatype   
_Default behavior without flag:_
  * A committed datatype in the source will be copied to the destination as a committed datatype. 
  * If a dataset in the source uses a committed datatype or an object in the source has an attribute of a committed datatype, that committed datatype will be written to the destination as an anonymous committed datatype. If copied in a single [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") operation, objects that share a committed datatype in the source will share an anonymous committed datatype in the destination copy. Subsequent [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") operations, however, will be unaware of prior anonymous committed datatypes and will create new ones.

See the “See Also” section immediately below for functions related to the use of this flag.  

See also
    Functions and a callback function used to tune committed datatype copying behavior:   
  * [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb)
  * [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...")
  * [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.")
  * [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.")
  * [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")
  * [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")
  * [Copying Committed Datatypes with H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/copying_committed.html)



Version
    1.8.9 [H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a11355b69cd246ae15e7d4c53cd210bd8) added in this release. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1)H5Pset_mcdt_search_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_mcdt_search_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb) |  _func_ ,   
|  | void * |  _op_data_ )  
Sets the callback function that [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will invoke before searching the entire destination file for a matching committed datatype.  

Parameters
     [in] | plist_id | Object copy property list identifier   
---|---|---  
[in] | func | User-defined callback function   
[in] | op_data | User-defined input data for the callback function 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...") allows an application to set a callback function, `func`, that will be invoked before searching the destination file for a matching committed datatype. The default, global search process is described in [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...").
The callback function must conform to the [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb) prototype and will return an instruction for one of the following actions:
  * Continue the search for a matching committed datatype in the destination file. 
  * Discontinue the search for a matching committed datatype. [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") will then apply the default behavior of creating an anonymous committed datatype. 
  * Abort the copy operation and exit [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.").


**Motivation:** [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...") provides the means to define a callback function. An application can then use that callback to take an additional action before the default search of all committed datatypes in the destination file or to take an action that replaces the default search. This mechanism is intended to improve performance when the global search might take a long time.
**Example** **Usage:** This example defines a callback function in the object copy property list.
```
static H5O_mcdt_search_ret_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a500f24019516869a4a61ac7efcb3b29b)
mcdt_search_cb(void *_udata)
{
    H5O_mcdt_search_ret_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a500f24019516869a4a61ac7efcb3b29b) action = *((H5O_mcdt_search_ret_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a500f24019516869a4a61ac7efcb3b29b) *)_udata);

     return(action);
 }

 int main(void) {
     hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);

     H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
     H5Padd_merge_committed_dtype_path(ocpypl_id, "/group/committed_dtypeA");

     action = H5O_MCDT_SEARCH_STOP;
     H5Pset_mcdt_search_cb(ocpypl_id, mcdt_search_cb, &action);
     H5Ocopy(...ocpypl_id...);
     ...
     ...
}

```


Note
     [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...") will fail if the object copy property list is invalid. 

Warning
    If the callback function return value causes [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") to abort, the destination file may be left in an inconsistent or corrupted state. 

See also
    
  * [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")
  * [H5O_mcdt_search_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#aa35611ba3daa73768194274c0e2a20eb)
  * [H5Padd_merge_committed_dtype_path()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#gab89c9debe50afca848151ff046afc82f "Adds a path to the list of paths that will be searched in the destination file for a matching committ...")
  * [H5Pfree_merge_committed_dtype_paths()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga4b8d6496ac56b167dba16729a8bd7adf "Clears the list of paths stored in the object copy property list.")
  * [H5Pget_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga5d7b82394d37bda28769a0435300d396 "Retrieves the callback function from the specified object copy property list.")
  * [H5Pset_copy_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga8819261e0b4663827212892e10dfc8a6 "Sets properties to be used when an object is copied.")
  * [H5Pset_mcdt_search_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html#ga9e0448885990a1b9ebd4493b7604f0c1 "Sets the callback function that H5Ocopy\(\) will invoke before searching the entire destination file fo...")
  * [Copying Committed Datatypes with H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/copying_committed.html)



Since
    1.8.9 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


