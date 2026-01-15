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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title2 "H2")
  * [◆ H5Fget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title3 "H2")
  * [◆ H5Fget_mdc_hit_rate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title4 "H2")
  * [◆ H5Fget_mdc_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title5 "H2")
  * [◆ H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title6 "H2")
  * [◆ H5Fget_mdc_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title7 "H2")
  * [◆ H5Freset_mdc_hit_rate_stats()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title8 "H2")
  * [◆ H5Fset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title9 "H2")
  * [◆ H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title10 "H2")
  * [◆ H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#func-members)
Metadata Cache
[Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) *config_ptr)  
| Obtains current metadata cache configuration for target file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mdc_hit_rate](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, double *hit_rate_ptr)  
| Obtains target file's metadata cache hit rate.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mdc_image_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *image_addr, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *image_size)  
| Obtains information about a cache image if it exists.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mdc_logging_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, bool *is_enabled, bool *is_currently_logging)  
| Gets the current metadata cache logging status.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fget_mdc_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, size_t *max_size_ptr, size_t *min_clean_size_ptr, size_t *cur_size_ptr, int *cur_num_entries_ptr)  
| Obtains current metadata cache size data for specified file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Freset_mdc_hit_rate_stats](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id)  
| Resets hit rate statistics counters for the target file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, const [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) *config_ptr)  
| Attempts to configure metadata cache of target file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fstart_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id)  
| Starts logging metadata cache events if logging was previously enabled.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fstop_mdc_logging](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id)  
| Stops logging metadata cache events if logging was previously enabled and is currently ongoing.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59)H5Fget_mdc_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mdc_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  |  [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) * |  _config_ptr_ )  
Obtains current metadata cache configuration for target file.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[in,out] | config_ptr | Pointer to the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) instance in which the current metadata cache configuration is to be reported. The fields of this structure are discussed [here](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_c-cache-config-t.html).  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Note
    The `in` direction applies only to the [H5AC_cache_config_t::version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f) field. All other fields are out parameters.  
[H5Fget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gaa67f127242d4aaf244ae8ac4a1fe6a59 "Obtains current metadata cache configuration for target file.") loads the current metadata cache configuration into the instance of [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) pointed to by the `config_ptr` parameter.  
The fields of the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) structure are shown below: 
typedef struct [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) {
/* general configuration fields: */
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [rpt_fcn_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af6bef1101778934a0f7d54963db4a9f5);
bool [open_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae92fb5b442027a53da62a853b9be7636);
bool [close_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ab047662ba4d5dd6eb32835b472c38fb0);
char [trace_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a92e7d20eb2b7b353961c64558ddac080)[[H5AC__MAX_TRACE_FILE_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_cpublic_8h.html#a717f1f3545cfc3d1b2208c96cc0c3bd3) + 1];
bool [evictions_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a1b7d261ea3c2cd28446253e8f3b67e0a);
bool [set_initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5bfa4ced2bee62567910bb14dd28265b);
size_t [initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a649236e7dd714855a50f122aa5caca9f);
double [min_clean_fraction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#abd805b98f873c1720f34a0ce937838fd);
size_t [max_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af4728438dee601cb2554d9bf18d78a43);
size_t [min_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af99ca22b80e05fd5b3603806348ab647);
long int [epoch_length](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac998e51b01e0eef09d9a29c43f97e4bf);
/* size increase control fields: */
enum [H5C_cache_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a040d488146ff1ca0a82209e9af3918fa) [incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae825aaf759060239e92170d20eb97d26);
double [lower_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a20f323fcb4747fc7228d2d74bb965586);
double [increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac504dff76b24ab9f15536c51aec9fbbb);
bool [apply_max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aef1ae6503a0ae34abacce69d2996628e);
size_t [max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ad5a729f1d611f2780679a35b3524052c);
enum [H5C_cache_flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#aaaa13ca7756d135b7df6d5a6779ee908) [flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a0e25a1dc2c695bea335df0e23ed6363c);
double [flash_multiple](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a77b1812e0407c9122db524462a5c9633);
double [flash_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a95fb1e03a77ef5c109d0c851416ced55);
/* size decrease control fields: */
enum [H5C_cache_decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a4f8534794ad9a977185a5d608c0af04f) [decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5df68196b281c19d8ab7da0788566aec);
double [upper_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a84a5ff4ac69196aa27c14f6f796db596);
double [decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a54007d3f2afb718b437f499a5c8b46d9);
bool [apply_max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ef00e6aab4e781e1e9b1344b3e23460);
size_t [max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a75e875a61c9da7f82482d0f6fe6e7152);
int [epochs_before_eviction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ac41e345300bdecd9943e855d55b71b);
bool [apply_empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a14cfe890a5a58b493084730c988aa4ec);
double [empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a9c1ae995513b55737aad09e11beff733);
/* parallel configuration fields: */
size_t [dirty_bytes_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a8e3c2a2d300b7a8f8d3705fc5e59a3c1);
int [metadata_write_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a83a536128dbb7785b2553c294f33d1fe);
} [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html);
(Click on a enumerator, field, or type for more information.) 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f)H5Fget_mdc_hit_rate()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mdc_hit_rate  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | double * |  _hit_rate_ptr_ )  
Obtains target file's metadata cache hit rate.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | hit_rate_ptr | Pointer to the double in which the hit rate is returned. Note that `hit_rate_ptr` is undefined if the API call fails  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fget_mdc_hit_rate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gabea066c3fd924d2cf868ecee66a7c41f "Obtains target file's metadata cache hit rate.") queries the metadata cache of the target file to obtain its hit rate `(cache hits / (cache hits + cache misses))` since the last time hit rate statistics were reset. If the cache has not been accessed since the last time the hit rate stats were reset, the hit rate is defined to be 0.0.
The hit rate stats can be reset either manually (via [H5Freset_mdc_hit_rate_stats()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f "Resets hit rate statistics counters for the target file.")), or automatically. If the cache's adaptive resize code is enabled, the hit rate stats will be reset once per epoch. If they are reset manually as well, the cache may behave oddly.
See the overview of the metadata cache in the special topics section of the user manual for details on the metadata cache and its adaptive resize algorithms. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a)H5Fget_mdc_image_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mdc_image_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  |  [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _image_addr_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _image_size_ )  
Obtains information about a cache image if it exists.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | image_addr | Offset of the cache image if it exists, or [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c) if it does not   
[out] | image_size | Length of the cache image if it exists, or 0 if it does not  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fget_mdc_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a "Obtains information about a cache image if it exists.") returns information about a cache image if it exists.
When an HDF5 file is opened in Read/Write mode, any metadata cache image will be read and deleted from the file on the first metadata cache access (or, if persistent free space managers are enabled, on the first file space allocation / deallocation, or read of free space manager status, whichever comes first).
Thus, if the file is opened Read/Write, [H5Fget_mdc_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a "Obtains information about a cache image if it exists.") should be called immediately after file open and before any other operation. If [H5Fget_mdc_image_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga7b37da15ff80c4aa5c275649f1f45b0a "Obtains information about a cache image if it exists.") is called after the cache image is loaded, it will correctly report that no cache image exists, as the image will have already been read and deleted from the file. In the Read Only case, the function may be called at any time, as any cache image will not be deleted from the file.  

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6)H5Fget_mdc_logging_status()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mdc_logging_status  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | bool * |  _is_enabled_ ,   
|  | bool * |  _is_currently_logging_ )  
Gets the current metadata cache logging status.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | is_enabled | Whether logging is enabled   
[out] | is_currently_logging | Whether events are currently being logged  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The metadata cache is a central part of the HDF5 library through which all _file metadata_ reads and writes take place. File metadata is normally invisible to the user and is used by the library for purposes such as locating and indexing data. File metadata should not be confused with user metadata, which consists of attributes created by users and attached to HDF5 objects such as datasets via H5A API calls.
Due to the complexity of the cache, a trace/logging feature has been created that can be used by HDF5 developers for debugging and performance analysis. The functions that control this functionality will normally be of use to a very limited number of developers outside of The HDF Group. The functions have been documented to help users create logs that can be sent with bug reports.
Control of the log functionality is straightforward. Logging is enabled via the [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.") function, which will modify the file access property list used to open or create a file. This function has a flag that determines whether logging begins at file open or starts in a paused state. Log messages can then be controlled via the [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") and [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") functions. [H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") can be used to examine a file access property list, and [H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") will return the current state of the logging flags.
The log format is described in the [Design: Metadata Cache Logging](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/Design-MetadataCache-Logging-THG20140224-v4.pdf) document. 

Note
    Unlike [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") and [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing."), this function can be called on any open file identifier. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f)H5Fget_mdc_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fget_mdc_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | size_t * |  _max_size_ptr_ ,   
|  | size_t * |  _min_clean_size_ptr_ ,   
|  | size_t * |  _cur_size_ptr_ ,   
|  | int * |  _cur_num_entries_ptr_ )  
Obtains current metadata cache size data for specified file.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[out] | max_size_ptr | Pointer to the location in which the current cache maximum size is to be returned, or NULL if this datum is not desired   
[out] | min_clean_size_ptr | Pointer to the location in which the current cache minimum clean size is to be returned, or NULL if that datum is not desired   
[out] | cur_size_ptr | Pointer to the location in which the current cache size is to be returned, or NULL if that datum is not desired   
[out] | cur_num_entries_ptr | Pointer to the location in which the current number of entries in the cache is to be returned, or NULL if that datum is not desired  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fget_mdc_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#gacda6cbd60d3c50b59f801eba4e5a335f "Obtains current metadata cache size data for specified file.") queries the metadata cache of the target file for the desired size information, and returns this information in the locations indicated by the pointer parameters. If any pointer parameter is NULL, the associated data is not returned.
If the API call fails, the values returned via the pointer parameters are undefined.
If adaptive cache resizing is enabled, the cache maximum size and minimum clean size may change at the end of each epoch. Current size and current number of entries can change on each cache access.
Current size can exceed maximum size under certain conditions. See the overview of the metadata cache in the special topics section of the user manual for a discussion of this. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f)H5Freset_mdc_hit_rate_stats()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Freset_mdc_hit_rate_stats  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _file_id_ | ) |   
---|---|---|---|---|---  
Resets hit rate statistics counters for the target file.  

Parameters
     [in] | file_id | File identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Freset_mdc_hit_rate_stats()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga6708886c2bb8740327d9078d7840197f "Resets hit rate statistics counters for the target file.") resets the hit rate statistics counters in the metadata cache associated with the specified file.
If the adaptive cache resizing code is enabled, the hit rate statistics are reset at the beginning of each epoch. This API call allows you to do the same thing from your program.
The adaptive cache resizing code may behave oddly if you use this call when adaptive cache resizing is enabled. However, the call should be useful if you choose to control metadata cache size from your program.
See [Metadata Caching in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n_m_d_c.html) for details about the metadata cache and the adaptive cache resizing algorithms. If you have not read, understood, and thought about the material covered in that documentation, you should not be using this API call.  

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8)H5Fset_mdc_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fset_mdc_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | const [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) * |  _config_ptr_ )  
Attempts to configure metadata cache of target file.  

Parameters
     [in] | file_id | File identifier   
---|---|---  
[in,out] | config_ptr | Pointer to the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) instance containing the desired configuration. The fields of this structure are discussed [here](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_c-cache-config-t.html).  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Fset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga81bc06be69131484eb04d01511b9c8f8 "Attempts to configure metadata cache of target file.") attempts to configure the file's metadata cache according configuration supplied in `config_ptr`. 
typedef struct [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) {
/* general configuration fields: */
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [rpt_fcn_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af6bef1101778934a0f7d54963db4a9f5);
bool [open_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae92fb5b442027a53da62a853b9be7636);
bool [close_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ab047662ba4d5dd6eb32835b472c38fb0);
char [trace_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a92e7d20eb2b7b353961c64558ddac080)[[H5AC__MAX_TRACE_FILE_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_cpublic_8h.html#a717f1f3545cfc3d1b2208c96cc0c3bd3) + 1];
bool [evictions_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a1b7d261ea3c2cd28446253e8f3b67e0a);
bool [set_initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5bfa4ced2bee62567910bb14dd28265b);
size_t [initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a649236e7dd714855a50f122aa5caca9f);
double [min_clean_fraction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#abd805b98f873c1720f34a0ce937838fd);
size_t [max_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af4728438dee601cb2554d9bf18d78a43);
size_t [min_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af99ca22b80e05fd5b3603806348ab647);
long int [epoch_length](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac998e51b01e0eef09d9a29c43f97e4bf);
/* size increase control fields: */
enum [H5C_cache_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a040d488146ff1ca0a82209e9af3918fa) [incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae825aaf759060239e92170d20eb97d26);
double [lower_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a20f323fcb4747fc7228d2d74bb965586);
double [increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac504dff76b24ab9f15536c51aec9fbbb);
bool [apply_max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aef1ae6503a0ae34abacce69d2996628e);
size_t [max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ad5a729f1d611f2780679a35b3524052c);
enum [H5C_cache_flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#aaaa13ca7756d135b7df6d5a6779ee908) [flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a0e25a1dc2c695bea335df0e23ed6363c);
double [flash_multiple](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a77b1812e0407c9122db524462a5c9633);
double [flash_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a95fb1e03a77ef5c109d0c851416ced55);
/* size decrease control fields: */
enum [H5C_cache_decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a4f8534794ad9a977185a5d608c0af04f) [decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5df68196b281c19d8ab7da0788566aec);
double [upper_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a84a5ff4ac69196aa27c14f6f796db596);
double [decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a54007d3f2afb718b437f499a5c8b46d9);
bool [apply_max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ef00e6aab4e781e1e9b1344b3e23460);
size_t [max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a75e875a61c9da7f82482d0f6fe6e7152);
int [epochs_before_eviction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ac41e345300bdecd9943e855d55b71b);
bool [apply_empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a14cfe890a5a58b493084730c988aa4ec);
double [empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a9c1ae995513b55737aad09e11beff733);
/* parallel configuration fields: */
size_t [dirty_bytes_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a8e3c2a2d300b7a8f8d3705fc5e59a3c1);
int [metadata_write_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a83a536128dbb7785b2553c294f33d1fe);
} [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html);
(Click on a enumerator, field, or type for more information.) 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67)H5Fstart_mdc_logging()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fstart_mdc_logging  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _file_id_ | ) |   
---|---|---|---|---|---  
Starts logging metadata cache events if logging was previously enabled.  

Parameters
     [in] | file_id | File identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The metadata cache is a central part of the HDF5 library through which all _file metadata_ reads and writes take place. File metadata is normally invisible to the user and is used by the library for purposes such as locating and indexing data. File metadata should not be confused with user metadata, which consists of attributes created by users and attached to HDF5 objects such as datasets via H5A API calls.
Due to the complexity of the cache, a trace/logging feature has been created that can be used by HDF5 developers for debugging and performance analysis. The functions that control this functionality will normally be of use to a very limited number of developers outside of The HDF Group. The functions have been documented to help users create logs that can be sent with bug reports.
Control of the log functionality is straightforward. Logging is enabled via the [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.") function, which will modify the file access property list used to open or create a file. This function has a flag that determines whether logging begins at file open or starts in a paused state. Log messages can then be controlled via the [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") and [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") functions. [H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") can be used to examine a file access property list, and [H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") will return the current state of the logging flags.
The log format is described in the [Design: Metadata Cache Logging](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/Design-MetadataCache-Logging-THG20140224-v4.pdf) document. 

Note
    Logging can only be started or stopped if metadata cache logging was enabled via [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.").  
When enabled and currently logging, the overhead of the logging feature will almost certainly be significant.  
The log file is opened when the HDF5 file is opened or created and not when this function is called for the first time.  
This function opens the log file and starts logging metadata cache operations for a particular file. Calling this function when logging has already been enabled will be considered an error. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234)H5Fstop_mdc_logging()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fstop_mdc_logging  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _file_id_ | ) |   
---|---|---|---|---|---  
Stops logging metadata cache events if logging was previously enabled and is currently ongoing.  

Parameters
     [in] | file_id | File identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The metadata cache is a central part of the HDF5 library through which all _file metadata_ reads and writes take place. File metadata is normally invisible to the user and is used by the library for purposes such as locating and indexing data. File metadata should not be confused with user metadata, which consists of attributes created by users and attached to HDF5 objects such as datasets via H5A API calls.
Due to the complexity of the cache, a trace/logging feature has been created that can be used by HDF5 developers for debugging and performance analysis. The functions that control this functionality will normally be of use to a very limited number of developers outside of The HDF Group. The functions have been documented to help users create logs that can be sent with bug reports.
Control of the log functionality is straightforward. Logging is enabled via the [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.") function, which will modify the file access property list used to open or create a file. This function has a flag that determines whether logging begins at file open or starts in a paused state. Log messages can then be controlled via the [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") and [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") functions. [H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") can be used to examine a file access property list, and [H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") will return the current state of the logging flags.
The log format is described in the [Design: Metadata Cache Logging](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/Design-MetadataCache-Logging-THG20140224-v4.pdf) document. 

Note
    Logging can only be started or stopped if metadata cache logging was enabled via [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.").  
This function only suspends the logging operations. The log file will remain open and will not be closed until the HDF5 file is closed. 

Since
    1.10.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


