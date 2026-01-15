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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title2 "H2")
  * [◆ H5Lis_registered()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title3 "H2")
  * [◆ H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title4 "H2")
  * [◆ H5Lunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#func-members)
Advanced Link Functions
[Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html)
## Detailed Description
Registration of User-defined links 
##  Functions  
---  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2) ([H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) id)  
| Determines whether a class of user-defined links is registered.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f) (const [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html) *cls)  
| Registers a user-defined link class or changes behavior of an existing class.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec) ([H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) id)  
| Unregisters a class of user-defined links.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2)H5Lis_registered()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Lis_registered  | ( | [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) | _id_ | ) |   
---|---|---|---|---|---  
Determines whether a class of user-defined links is registered.  

Parameters
     [in] | id | User-defined link class identifier  
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5Lis_registered()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2 "Determines whether a class of user-defined links is registered.") tests whether a user-defined link class is currently registered, either by the HDF5 library or by the user through the use of [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class."). 

Note
    A link class must be registered to create new links of that type or to traverse existing links of that type. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f)H5Lregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lregister  | ( | const [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html) * | _cls_ | ) |   
---|---|---|---|---|---  
Registers a user-defined link class or changes behavior of an existing class.  

Parameters
     [in] | cls | Pointer to a buffer containing the struct describing the user-defined link class  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class.") registers a class of user-defined links, or changes the behavior of an existing class.
`cls` is a pointer to a buffer containing a copy of the [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html "Link prototype.") struct. This struct is defined in [H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html) as follows: 
typedef struct {
int version; 
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) id; 
const char *comment; 
[H5L_create_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a35f5eeac9ed28bfcc94d5037cac5af4a) create_func; 
[H5L_move_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a923e0abe07eacf3101ba6e717665edf1) move_func; 
[H5L_copy_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a5800df80f7709b30fa99dfc5166468e5) copy_func; 
[H5L_traverse_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a7daba3925a1b3b28a1944788d2a88597) trav_func; 
[H5L_delete_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a4ef5c91e382760e74fb95a480f0773ca) del_func; 
[H5L_query_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#a4ed3f3e89fe170d547cefd7d68549632) query_func; 
} [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html);
The class definition passed with `cls` must include at least the following: 
  * An [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html "Link prototype.") version (which should be [H5L_LINK_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ldevelop_8h.html#ac4ef431ffac7939c7d37a2e55e49cd63 "Current version of the H5L_class_t struct.")) 
  * A link class identifier, `class_id`
  * A traversal function, `trav_func`


Remaining `struct` members are optional and may be passed as NULL.
The link class passed in `class_id` must be in the user-definable range between [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f) and [H5L_TYPE_UD_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a6f9109ae61b6bd1b6668237cce13c75f) (see the table below) and will override any existing link class with that identifier.
As distributed, valid values of `class_id` used in HDF5 include the following (defined in [H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html)): 
[H5L_TYPE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48ab90f13082490fcd293a228b2785489e3) | Hard link  
---|---  
[H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed) | Soft link  
[H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) | External link  
[H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d) | Error  
The hard and soft link class identifiers cannot be modified or reassigned, but the external link class is implemented as an example in the user-definable link class identifier range. [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class.") is used to register additional link classes. It could also be used to modify the behavior of the external link class, though that is not recommended.
The following table summarizes existing link types and values and the reserved and user-definable link class identifier value ranges. 
Link class identifier or Value range  | Description  | Link class or label   
---|---|---  
0 to 63  | Reserved range  |   
64 to 255  | User-definable range  |   
64  | Minimum user-defined value  |  [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f)  
64  | External link  |  [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06)  
255  | Maximum user-defined value  |  [H5L_TYPE_UD_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a6f9109ae61b6bd1b6668237cce13c75f)  
255  | Maximum value  |  [H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd)  
-1  | Error  |  [H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d)  
Note that HDF5 internally registers user-defined link classes only by the numeric value of the link class identifier. An application, on the other hand, will generally use a name for a user-defined class, if for no other purpose than as a variable name. Assume, for example, that a complex link type is registered with the link class identifier 73 and that the code includes the following assignment: 
H5L_TYPE_COMPLEX_A = 73
The application can refer to the link class with a term, `H5L_TYPE_COMPLEX_A`, that conveys meaning to a human reviewing the code, while HDF5 recognizes it by the more cryptic numeric identifier, 73. 

Attention
    Important details and considerations include the following: 
  * If you plan to distribute files or software with a user-defined link class, please contact the Help Desk at The HDF Group to help prevent collisions between `class_id` values. See below. 
  * As distributed with HDF5, the external link class is implemented as an example of a user-defined link class with [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) equal to [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f). `class_id` in the [H5L_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__class__t.html "Link prototype.") `struct` must not equal [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f) unless you intend to overwrite or modify the behavior of external links. 
  * [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class.") can be used only with link class identifiers in the user-definable range (see table above). 
  * The hard and soft links defined by the HDF5 library, [H5L_TYPE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48ab90f13082490fcd293a228b2785489e3) and [H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed), reside in the reserved range below [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f) and cannot be redefined or modified. 
  * [H5Lis_registered()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2 "Determines whether a class of user-defined links is registered.") can be used to determine whether a desired link class identifier is available. _Note that this function will tell you only whether the link class identifier has been registered with the installed copy of HDF5; it cannot tell you whether the link class has been registered with The HDF Group._
  * [H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd) is the maximum allowed value for a link type identifier. 
  * [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f) equals [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06). 
  * [H5L_TYPE_UD_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a6f9109ae61b6bd1b6668237cce13c75f) equals [H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd). 
  * [H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d) indicates that an error has occurred.



Note
     **Registration with The HDF Group:**  
There are sometimes reasons to take a broader approach to registering a user-defined link class than just invoking [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class."). For example: 
  * A user-defined link class is intended for use across an organization, among collaborators, or across a community of users. 
  * An application or library overlying HDF5 invokes a user-defined link class that must be shipped with the software. 
  * Files are distributed that make use of a user-defined link class. 
  * Or simply, a specific user-defined link class is thought to be widely useful.

In such cases, you are encouraged to register that link class with The HDF Group's Helpdesk. The HDF Group maintains a registry of known user-defined link classes and tracks the selected link class identifiers. This registry is intended to reduce the risk of collisions between `class_id` values and to help coordinate the use of specialized link classes. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec)H5Lunregister()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lunregister  | ( | [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) | _id_ | ) |   
---|---|---|---|---|---  
Unregisters a class of user-defined links.  

Parameters
     [in] | id | User-defined link class identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec "Unregisters a class of user-defined links.") unregisters a class of user-defined links, preventing them from being traversed, queried, moved, etc. 

Note
    A link class can be re-registered using [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class."). 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


