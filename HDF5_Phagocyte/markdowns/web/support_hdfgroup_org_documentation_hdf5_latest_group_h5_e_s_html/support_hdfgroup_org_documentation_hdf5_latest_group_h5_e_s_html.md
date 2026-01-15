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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title2 "H2")
  * [◆ H5EScancel()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title3 "H2")
  * [◆ H5ESclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title4 "H2")
  * [◆ H5EScreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title5 "H2")
  * [◆ H5ESfree_err_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title6 "H2")
  * [◆ H5ESget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title7 "H2")
  * [◆ H5ESget_err_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title8 "H2")
  * [◆ H5ESget_err_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title9 "H2")
  * [◆ H5ESget_err_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title10 "H2")
  * [◆ H5ESget_op_counter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title11 "H2")
  * [◆ H5ESregister_complete_func()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title12 "H2")
  * [◆ H5ESregister_insert_func()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title13 "H2")
  * [◆ H5ESwait()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#func-members)
Event Set Interface (H5ES)
## Detailed Description
Event Set Interface.  

**[Todo](https://support.hdfgroup.org/documentation/hdf5/latest/todo.html#_todo000005)**
    Add the event set life cycle.
**This interface can only be used with the HDF5 VOL connectors that enable the asynchronous feature in HDF5.** The native HDF5 library has only synchronous operations.
HDF5 VOL connectors with support for asynchronous operations:
  * ASYNC
  * DAOS



Example:
    
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(..);
gid = [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)(fid, ..); //Starts when H5Fopen completes
did = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(gid, ..); //Starts when H5Gopen completes
es_id = [H5EScreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc)(); // Create event set for tracking async operations
status = [H5Dwrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7773f0c3503418421bcb535a95ee832e)(did, .., es_id); //Asynchronous, starts when H5Dopen completes,
// may run concurrently with other H5Dwrite_async
// in event set.
status = [H5Dwrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7773f0c3503418421bcb535a95ee832e)(did, .., es_id); //Asynchronous, starts when H5Dopen completes,
// may run concurrently with other H5Dwrite_async
// in event set....
<other user code>
...
H5ESwait(es_id); // Wait for operations in event set to complete, buffers
// used for H5Dwrite_async must only be changed after wait
// returns.
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Dwrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7773f0c3503418421bcb535a95ee832e)
herr_t H5Dwrite_async(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf, hid_t es_id)
[H5EScreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc)
hid_t H5EScreate(void)
Creates an event set.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
[H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)
#define H5Gopen
**Definition** H5version.h:1251
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5EScancel](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga31adb1cd449abdf71584af89195bf18e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, size_t *num_not_canceled, bool *err_occurred)  
| Attempt to cancel operations in an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1159e2c6748f200857dffa55011ae060) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
| Terminates access to an event set.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5EScreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc) (void)  
| Creates an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESfree_err_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gadaab9ec7ce5725bed34636e0cf8cad8d) (size_t num_err_info, [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) err_info[])  
| Convenience routine to free an array of [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) structs.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESget_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1a5a9212958bf7054a56107587d480d2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, size_t *count)  
| Retrieves number of events in an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESget_err_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga42b502d05e0dff40ec550c85bb54ca1c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, size_t *num_errs)  
| Retrieves the number of failed operations.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESget_err_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga7fe438ffe2d7fcede25e4bec5194c923) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, size_t num_err_info, [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) err_info[], size_t *err_cleared)  
| Retrieves information about failed operations.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESget_err_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga3f1d63eb93c8466e0132303f506f6d33) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, bool *err_occurred)  
| Checks for failed operations.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESget_op_counter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1880fb2353f677ef56a6204706cec4d0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, uint64_t *counter)  
| Retrieves the accumulative operation counter for an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESregister_complete_func](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gaad19a0bf9ef8816d178f3ee6d1d1ef50) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, [H5ES_event_complete_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#a203f983be81866b1d8e98c9b4790125c) func, void *ctx)  
| Registers a callback to invoke when an operation completes within an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESregister_insert_func](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gac2255859e4038a02e8dea94b2585e9e8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, [H5ES_event_insert_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#a0657307c8919bf968bd0fc8d7d6afa2d) func, void *ctx)  
| Registers a callback to invoke when a new operation is inserted into an event set.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5ESwait](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga4f125d5ee5e59e3f6005f07723d306aa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id, uint64_t timeout, size_t *num_in_progress, bool *err_occurred)  
| Waits for operations in event set to complete.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga31adb1cd449abdf71584af89195bf18e)H5EScancel()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5EScancel  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | size_t * |  _num_not_canceled_ ,   
|  | bool * |  _err_occurred_ )  
Attempt to cancel operations in an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[out] | num_not_canceled | The number of events not canceled   
[out] | err_occurred | Status indicating if error is present in the event set  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5EScancel()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga31adb1cd449abdf71584af89195bf18e "Attempt to cancel operations in an event set.") attempts to cancel operations in an event set specified by `es_id`. H5ES_NONE is a valid value for `es_id`, but functions as a no-op. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1159e2c6748f200857dffa55011ae060)H5ESclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _es_id_ | ) |   
---|---|---|---|---|---  
Terminates access to an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1159e2c6748f200857dffa55011ae060 "Terminates access to an event set.") terminates access to an event set specified by `es_id`. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc)H5EScreate()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5EScreate  | ( | void |  | ) |   
---|---|---|---|---|---  
Creates an event set.  

Returns
    Returns a event set identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).
[H5EScreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc "Creates an event set.") creates a new event set and returns a corresponding event set identifier. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gadaab9ec7ce5725bed34636e0cf8cad8d)H5ESfree_err_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESfree_err_info  | ( | size_t |  _num_err_info_ ,   
---|---|---|---  
|  | [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) |  _err_info_[] )  
Convenience routine to free an array of [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) structs.  

Parameters
     [in] | num_err_info | The number of elements in `err_info` array   
---|---|---  
[in] | err_info | Array of structures  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Since
    1.14.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1a5a9212958bf7054a56107587d480d2)H5ESget_count()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESget_count  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | size_t * |  _count_ )  
Retrieves number of events in an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[out] | count | The number of events in the event set  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1a5a9212958bf7054a56107587d480d2 "Retrieves number of events in an event set.") retrieves number of events in an event set specified by `es_id`. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga42b502d05e0dff40ec550c85bb54ca1c)H5ESget_err_count()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESget_err_count  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | size_t * |  _num_errs_ )  
Retrieves the number of failed operations.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[out] | num_errs | Number of errors  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESget_err_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga42b502d05e0dff40ec550c85bb54ca1c "Retrieves the number of failed operations.") retrieves the number of failed operations in an event set specified by `es_id`.
The function does not wait for active operations to complete, so count may not include all failures. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga7fe438ffe2d7fcede25e4bec5194c923)H5ESget_err_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESget_err_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | size_t |  _num_err_info_ ,   
|  | [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) |  _err_info_[],   
|  | size_t * |  _err_cleared_ )  
Retrieves information about failed operations.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[in] | num_err_info | The number of elements in `err_info` array   
[out] | err_info | Array of structures   
[out] | err_cleared | Number of cleared errors  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESget_err_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga7fe438ffe2d7fcede25e4bec5194c923 "Retrieves information about failed operations.") retrieves information about failed operations in an event set specified by `es_id`. The strings retrieved for each error info must be released by calling [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library.").
Below is the description of the [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) structure: 
typedef struct [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html) {
/* API call info */
char *[api_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a626231556035d07c144783eb7b69c7e0); 
char *[api_args](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a2e77d439e7cd1f696a0f3e828a04728f); 
/* Application info */
char *[app_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a783ef99ec8a1aedceea00489ebfffeb5); 
char *[app_func_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a13b087b084bb3c56b0d1a5c2e57e09f8); 
unsigned [app_line_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a5ca51c24a0c2c6f3b73c2a3b3a2282e8); 
/* Operation info */
uint64_t [op_ins_count](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a33675701c7fe6aad490a6b4e1ff99345); 
uint64_t [op_ins_ts](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#af5de851e0076fcf1b9d1db8f4231c174); 
uint64_t [op_exec_ts](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a84e50df54a166388156fc1d5775c213e); 
uint64_t [op_exec_time](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#a35b3ad1e8c729416bb5506bc6a09069a); 
/* Error info */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [err_stack_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html#ac717c89bff4c1eac778f760deec58c31); 
} [H5ES_err_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e_s__err__info__t.html);
(Click on a enumerator, field, or type for more information.) 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga3f1d63eb93c8466e0132303f506f6d33)H5ESget_err_status()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESget_err_status  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | bool * |  _err_occurred_ )  
Checks for failed operations.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[out] | err_occurred | Status indicating if error is present in the event set  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESget_err_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga3f1d63eb93c8466e0132303f506f6d33 "Checks for failed operations.") checks if event set specified by es_id has failed operations. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1880fb2353f677ef56a6204706cec4d0)H5ESget_op_counter()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESget_op_counter  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | uint64_t * |  _counter_ )  
Retrieves the accumulative operation counter for an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[out] | counter | The accumulative counter value for an event set  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESget_op_counter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1880fb2353f677ef56a6204706cec4d0 "Retrieves the accumulative operation counter for an event set.") retrieves the current accumulative count of event set operations since the event set creation of `es_id`. 

Note
    This is designed for wrapper libraries mainly, to use as a mechanism for matching operations inserted into the event set with possible errors that occur. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gaad19a0bf9ef8816d178f3ee6d1d1ef50)H5ESregister_complete_func()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESregister_complete_func  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | [H5ES_event_complete_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#a203f983be81866b1d8e98c9b4790125c) |  _func_ ,   
|  | void * |  _ctx_ )  
Registers a callback to invoke when an operation completes within an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[in] | func | The completion function to register   
[in] | ctx | User-specified information (context) to pass to `func` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Only one complete callback can be registered for each event set. Registering a new callback will replace the existing one. H5ES_NONE is a valid value for 'es_id', but functions as a no-op 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gac2255859e4038a02e8dea94b2585e9e8)H5ESregister_insert_func()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESregister_insert_func  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | [H5ES_event_insert_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_spublic_8h.html#a0657307c8919bf968bd0fc8d7d6afa2d) |  _func_ ,   
|  | void * |  _ctx_ )  
Registers a callback to invoke when a new operation is inserted into an event set.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[in] | func | The insert function to register   
[in] | ctx | User-specified information (context) to pass to `func` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Only one insert callback can be registered for each event set. Registering a new callback will replace the existing one. H5ES_NONE is a valid value for 'es_id', but functions as a no-op 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga4f125d5ee5e59e3f6005f07723d306aa)H5ESwait()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5ESwait  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ ,   
---|---|---|---  
|  | uint64_t |  _timeout_ ,   
|  | size_t * |  _num_in_progress_ ,   
|  | bool * |  _err_occurred_ )  
Waits for operations in event set to complete.  

Parameters
     [in] | es_id | Event set identifier   
---|---|---  
[in] | timeout | Total time in nanoseconds to wait for all operations in the event set to complete   
[out] | num_in_progress | The number of operations still in progress   
[out] | err_occurred | Flag if an operation in the event set failed  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5ESwait()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga4f125d5ee5e59e3f6005f07723d306aa "Waits for operations in event set to complete.") waits for operations in an event set `es_id` to wait with `timeout`.
Timeout value is in nanoseconds, and is for the [H5ESwait()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga4f125d5ee5e59e3f6005f07723d306aa "Waits for operations in event set to complete.") call and not for each individual operation in the event set. For example, if "10" is passed as a timeout value and the event set waited 4 nanoseconds for the first operation to complete, the remaining operations would be allowed to wait for at most 6 nanoseconds more, i.e., the timeout value used across all operations in the event set until it reaches 0, then any remaining operations are only checked for completion, not waited on.
This call will stop waiting on operations and will return immediately if an operation fails. If a failure occurs, the value returned for the number of operations in progress may be inaccurate. 

Since
    1.14.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


