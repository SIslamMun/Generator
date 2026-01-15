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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title2 "H2")
  * [◆ H5Iclear_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title3 "H2")
  * [◆ H5Idec_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title4 "H2")
  * [◆ H5Idestroy_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title5 "H2")
  * [◆ H5Iget_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title6 "H2")
  * [◆ H5Iinc_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title7 "H2")
  * [◆ H5Iiterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title8 "H2")
  * [◆ H5Inmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title9 "H2")
  * [◆ H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title10 "H2")
  * [◆ H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title11 "H2")
  * [◆ H5Iregister_type1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title12 "H2")
  * [◆ H5Iregister_type2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title13 "H2")
  * [◆ H5Iremove_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title14 "H2")
  * [◆ H5Isearch()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title15 "H2")
  * [◆ H5Itype_exists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#title16 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#func-members)
User-defined ID Types
[Identifiers (H5I)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html)
## Detailed Description
The [Identifiers (H5I)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html) module contains functions to define new identifier types. For convenience, handles of type [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) can then be associated with the new identifier types and user objects.
New identifier types can be created by registering a new identifier type with the HDF5 library. Once a new identifier type has bee registered, it can be used to generate identifiers for user objects.
User-defined identifier types can be searched and iterated.
Like library-defined identifiers, user-defined identifiers _and_ identifier types are reference counted, and the reference counts can be manipulated accordingly.
User-defined identifiers no longer in use should be deleted or reclaimed, and identifier types should be destroyed if they are no longer required.
Create | Read   
---|---  
122 [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) free_func(void *obj) 123 { 124 printf("Calling free_func...\n"); 125 [H5free_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf)(obj); 126 return 0; 127 } 128 129 { 130 __label__ fail_id, fail_obj, fail_register; 131 [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type; 132 int *obj; 133 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id; 134 135 // register a new ID type 136 if ((type = [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)(1024, &free_func)) < 0) { 137 ret_val = EXIT_FAILURE; 138 goto fail_register; 139 } 140 printf("New identifier type ID: %d\n", type); 141 142 // create a new object 143 if ((obj = [H5allocate_memory](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga22628978a6ac53bee610ea5d22e48683)(10 * sizeof(int), 0)) == NULL) { 144 ret_val = EXIT_FAILURE; 145 goto fail_obj; 146 } 147 148 // obtain an ID for the new object 149 if ((obj_id = [H5Iregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5)(type, obj)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 150 ret_val = EXIT_FAILURE; 151 goto fail_id; 152 } 153 printf("New object identifier: %ld\n", obj_id); 154 155fail_id: 156 [H5Iclear_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7)(type, 1); 157fail_obj: 158 [H5Idestroy_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e)(type); 159fail_register:; 160 } |  164 { 165 __label__ fail_query, fail_register; 166 [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type; 167 [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) count; 168 169 // register a new ID type 170 if ((type = [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)(1024, NULL)) < 0) { 171 ret_val = EXIT_FAILURE; 172 goto fail_register; 173 } 174 printf("New FOO identifier type ID: %d\n", type); 175 176 // return the number of identifiers of TYPE currently in use 177 if ([H5Inmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga839170c17fb8be7e14d16ee74a314a33)(type, &count) < 0) { 178 ret_val = EXIT_FAILURE; 179 goto fail_query; 180 } 181 printf("%llu FOO identifiers in use\n", count); 182 183fail_query: 184 [H5Idestroy_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e)(type); 185fail_register:; 186 }  
Update | Delete   
190 { 191 __label__ fail_id, fail_register; 192 [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type; 193 int obj = 42; 194 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id; 195 196 // register a new ID type 197 if ((type = [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)(1024, NULL)) < 0) { 198 ret_val = EXIT_FAILURE; 199 goto fail_register; 200 } 201 printf("New identifier type ID: %d\n", type); 202 203 // obtain an ID for the new object 204 if ((obj_id = [H5Iregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5)(type, &obj)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 205 ret_val = EXIT_FAILURE; 206 goto fail_id; 207 } 208 printf("New object identifier: %ld\n", obj_id); 209 210 // bump the reference count 211 assert([H5Iinc_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f)(obj_id) == 2); 212 213 // force the deletion of all IDs regardless of reference count 214 [H5Iclear_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7)(type, 1); 215fail_id: 216 [H5Idestroy_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e)(type); 217fail_register:; 218 } |  222 { 223 __label__ fail_register; 224 [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type; 225 226 // register a new ID type 227 if ((type = [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)(1024, NULL)) < 0) { 228 ret_val = EXIT_FAILURE; 229 goto fail_register; 230 } 231 printf("New identifier type ID: %d\n", type); 232 233 // decrementing the identifier type's ref. count to 0 triggers 234 // the type to be destroyed 235 assert([H5Idec_type_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3d29394ea751712e8ef5f9548f611e57)(type) == 0); 236 237fail_register:; 238 }  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Iclear_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, bool force)  
| Deletes all identifiers of the given type.   
  
int  |  [H5Idec_type_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3d29394ea751712e8ef5f9548f611e57) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Decrements the reference count on an identifier type.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Idestroy_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Removes an identifier type and all identifiers within that type.   
  
int  |  [H5Iget_type_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga991d7305ffbd4fdd84432b58f704fd3b) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Retrieves the reference count on an ID type.   
  
int  |  [H5Iinc_type_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gae8302de9f2734882596a83f3117f2c9e) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Increments the reference count on an ID type.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Iiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, [H5I_iterate_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a23a008d105d19b53a0dd7640362727de) op, void *op_data)  
| Calls a callback for each member of the identifier type specified.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Inmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga839170c17fb8be7e14d16ee74a314a33) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *num_members)  
| Returns the number of identifiers in a given identifier type.   
  
void *  |  [H5Iobject_verify](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Returns the object referenced by an ID.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Iregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, const void *object)  
| Registers an object under a type and returns an ID for it.   
  
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  [H5Iregister_type1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab57559f14a16d4375815e45054abad16) (size_t hash_size, unsigned reserved, [H5I_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a5f6d1576913865b92856474bfedbe2b4) free_func)  
| Creates and returns a new ID type.   
  
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2) (unsigned reserved, [H5I_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a5f6d1576913865b92856474bfedbe2b4) free_func)  
| Creates and returns a new ID type.   
  
void *  |  [H5Iremove_verify](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Removes an ID from its type.   
  
void *  |  [H5Isearch](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga1f59a0d6e594ce7d61d85e1c16a0a381) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, [H5I_search_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a7400bdcded3eef3c4c4cf930173f9f72) func, void *key)  
| Finds the memory referred to by an ID within the given ID type such that some criterion is satisfied.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Itype_exists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gac137a599a2055909918e55955a69dfad) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type)  
| Determines whether an identifier type is registered.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7)H5Iclear_type()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Iclear_type  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  | bool |  _force_ )  
Deletes all identifiers of the given type.  

Parameters
     [in] | type | Identifier of identifier type which is to be cleared of identifiers   
---|---|---  
[in] | force | Whether or not to force deletion of all identifiers 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Iclear_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7 "Deletes all identifiers of the given type.") deletes all identifiers of the type identified by the argument `type`.
The identifier type's free function is first called on all of these identifiers to free their memory, then they are removed from the type.
If the `force` flag is set to false, only those identifiers whose reference counts are equal to 1 will be deleted, and all other identifiers will be entirely unchanged. If the force flag is true, all identifiers of this type will be deleted. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3d29394ea751712e8ef5f9548f611e57)H5Idec_type_ref()
int H5Idec_type_ref  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) | _type_ | ) |   
---|---|---|---|---|---  
Decrements the reference count on an identifier type.  

Parameters
     [in] | type | The identifier of the type whose reference count is to be decremented  
---|---|--- 

Returns
    Returns the current reference count on success, negative on failure.  
[H5Idec_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3d29394ea751712e8ef5f9548f611e57 "Decrements the reference count on an identifier type.") decrements the reference count on an identifier type. The reference count is used by the library to indicate when an identifier type can be destroyed. If the reference count reaches zero, this function will destroy it.
The type parameter is the identifier for the identifier type whose reference count is to be decremented. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e)H5Idestroy_type()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Idestroy_type  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) | _type_ | ) |   
---|---|---|---|---|---  
Removes an identifier type and all identifiers within that type.  

Parameters
     [in] | type | Identifier of identifier type which is to be destroyed  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
H5Idestroy_type deletes an entire identifier type `type`. All identifiers of this type are destroyed and no new identifiers of this type can be registered.
The type's free function is called on all of the identifiers which are deleted by this function, freeing their memory. In addition, all memory used by this type's hash table is freed.
Since the [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) values of destroyed identifier types are reused when new types are registered, it is a good idea to set the variable holding the value of the destroyed type to [H5I_UNINIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bac3bd281c14f862130b9dc14ceb8acbf0). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga991d7305ffbd4fdd84432b58f704fd3b)H5Iget_type_ref()
int H5Iget_type_ref  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) | _type_ | ) |   
---|---|---|---|---|---  
Retrieves the reference count on an ID type.  

Parameters
     [in] | type | The identifier of the type whose reference count is to be retrieved  
---|---|--- 

Returns
    Returns the current reference count on success, negative on failure.  
[H5Iget_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga991d7305ffbd4fdd84432b58f704fd3b "Retrieves the reference count on an ID type.") retrieves the reference count on an ID type. The reference count is used by the library to indicate when an ID type can be destroyed.
The type parameter is the identifier for the ID type whose reference count is to be retrieved. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gae8302de9f2734882596a83f3117f2c9e)H5Iinc_type_ref()
int H5Iinc_type_ref  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) | _type_ | ) |   
---|---|---|---|---|---  
Increments the reference count on an ID type.  

Parameters
     [in] | type | The identifier of the type whose reference count is to be incremented  
---|---|--- 

Returns
    Returns the current reference count on success, negative on failure.  
[H5Iinc_type_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gae8302de9f2734882596a83f3117f2c9e "Increments the reference count on an ID type.") increments the reference count on an ID type. The reference count is used by the library to indicate when an ID type can be destroyed.
The type parameter is the identifier for the ID type whose reference count is to be incremented. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4)H5Iiterate()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Iiterate  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  | [H5I_iterate_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a23a008d105d19b53a0dd7640362727de) |  _op_ ,   
|  | void * |  _op_data_ )  
Calls a callback for each member of the identifier type specified.  

Parameters
     [in] | type | The identifier type   
---|---|---  
[in] | op | The callback function   
[in,out] | op_data | The data for the callback function 

Returns
    The last value returned by `op`  
[H5Iiterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4 "Calls a callback for each member of the identifier type specified.") calls the callback function `op` for each member of the identifier type `type`. The callback function type for `op`, [H5I_iterate_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a23a008d105d19b53a0dd7640362727de), is defined in [H5Ipublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html) as: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5I_iterate_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a23a008d105d19b53a0dd7640362727de))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, void *udata);
`op` takes as parameters the identifier and a pass through of `op_data`, and returns an [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160).
A positive return from op will cause the iteration to stop and [H5Iiterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4 "Calls a callback for each member of the identifier type specified.") will return the value returned by `op`. A negative return from `op` will cause the iteration to stop and [H5Iiterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga3bc2a11a594bf484cc5dc69ac987d0f4 "Calls a callback for each member of the identifier type specified.") will return failure. A zero return from `op` will allow iteration to continue, as long as there are other identifiers remaining in type. 

Warning
    Adding or removing members of the identifier type during iteration will lead to undefined behavior. 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga839170c17fb8be7e14d16ee74a314a33)H5Inmembers()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Inmembers  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _num_members_ )  
Returns the number of identifiers in a given identifier type.  

Parameters
     [in] | type | The identifier type   
---|---|---  
[out] | num_members | Number of identifiers of the specified identifier type 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Inmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga839170c17fb8be7e14d16ee74a314a33 "Returns the number of identifiers in a given identifier type.") returns the number of identifiers of the identifier type specified in `type`.
The number of identifiers is returned in `num_members`. If no identifiers of this type have been registered, the type does not exist, or it has been destroyed, `num_members` is returned with the value 0. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035)H5Iobject_verify()
void * H5Iobject_verify  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _id_ ,   
---|---|---|---  
|  | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ )  
Returns the object referenced by an ID.  

Parameters
     [in] | id | ID to be dereferenced   
---|---|---  
[in] | type | The identifier type 

Returns
    Pointer to the object referenced by `id` on success, NULL on failure.  
[H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID.") returns a pointer to the memory referenced by id after verifying that `id` is of type `type`. This function is analogous to dereferencing a pointer in C with type checking. 

Note
     [H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID.") does not change the ID it is called on in any way (as opposed to [H5Iremove_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a "Removes an ID from its type."), which removes the ID from its type's hash table). 

See also
    [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5)H5Iregister()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Iregister  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  | const void * |  _object_ )  
Registers an object under a type and returns an ID for it.  

Parameters
     [in] | type | The identifier of the type of the new ID   
---|---|---  
[in] | object | Pointer to object for which a new ID is created 

Returns
    Returns a object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it.") creates and returns a new ID for an object.
The `type` parameter is the identifier for the ID type to which this new ID will belong. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab).
The `object` parameter is a pointer to the memory which the new ID will be a reference to. This pointer will be stored by the library and returned via a call to [H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID."). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab57559f14a16d4375815e45054abad16)H5Iregister_type1()
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) H5Iregister_type1  | ( | size_t |  _hash_size_ ,   
---|---|---|---  
|  | unsigned |  _reserved_ ,   
|  | [H5I_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a5f6d1576913865b92856474bfedbe2b4) |  _free_func_ )  
Creates and returns a new ID type.  

Parameters
     [in] | hash_size | Minimum hash table size (in entries) used to store IDs for the new type (unused in 1.8.13 and later)   
---|---|---  
[in] | reserved | Number of reserved IDs for the new type   
[in] | free_func | Function used to deallocate space for a single ID 

Returns
    Returns the type identifier on success, negative on failure.  
[H5Iregister_type1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab57559f14a16d4375815e45054abad16 "Creates and returns a new ID type.") allocates space for a new ID type and returns an identifier for it.
The `hash_size` parameter indicates the minimum size of the hash table used to store IDs in the new type. This parameter is unused in 1.8.13 and later, when the implementation of ID storage changed.
The `reserved` parameter indicates the number of IDs in this new type to be reserved. Reserved IDs are valid IDs which are not associated with any storage within the library.
The `free_func` parameter is a function pointer to a function which returns an [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) and accepts a `void*`. The purpose of this function is to deallocate memory for a single ID. It will be called by [H5Iclear_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7 "Deletes all identifiers of the given type.") and [H5Idestroy_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e "Removes an identifier type and all identifiers within that type.") on each ID. This function is NOT called by [H5Iremove_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a "Removes an ID from its type."). The `void*` will be the same pointer which was passed in to the [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it.") function. The `free_func` function should return 0 on success and -1 on failure. 

Since
    1.8.0  

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000058)**
    2.0.0 Deprecated in favor of the function [H5Iregister_type2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2 "Creates and returns a new ID type.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)H5Iregister_type2()
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) H5Iregister_type2  | ( | unsigned |  _reserved_ ,   
---|---|---|---  
|  | [H5I_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a5f6d1576913865b92856474bfedbe2b4) |  _free_func_ )  
Creates and returns a new ID type.  

Parameters
     [in] | reserved | Number of reserved IDs for the new type   
---|---|---  
[in] | free_func | Function used to deallocate space for a single ID 

Returns
    Returns the type identifier on success, negative on failure.  
[H5Iregister_type2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2 "Creates and returns a new ID type.") allocates space for a new ID type and returns an identifier for it.
The `reserved` parameter indicates the number of IDs in this new type to be reserved. Reserved IDs are valid IDs which are not associated with any storage within the library.
The `free_func` parameter is a function pointer to a function which returns an [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) and accepts a `void*`. The purpose of this function is to deallocate memory for a single ID. It will be called by [H5Iclear_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab019005507b8a02b16538b864dd094c7 "Deletes all identifiers of the given type.") and [H5Idestroy_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaabc75ed99fd6b206154b899fac64977e "Removes an identifier type and all identifiers within that type.") on each ID. This function is NOT called by [H5Iremove_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a "Removes an ID from its type."). The `void*` will be the same pointer which was passed in to the [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it.") function. The `free_func` function should return 0 on success and -1 on failure. 

Since
    2.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a)H5Iremove_verify()
void * H5Iremove_verify  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _id_ ,   
---|---|---|---  
|  | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ )  
Removes an ID from its type.  

Parameters
     [in] | id | The ID to be removed from its type   
---|---|---  
[in] | type | The identifier type 

Returns
    Returns a pointer to the memory referred to by `id` on success, NULL on failure.  
[H5Iremove_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gaa297aca26ed0383a6e126b92454c619a "Removes an ID from its type.") first ensures that `id` belongs to `type`. If so, it removes `id` from its type and returns the pointer to the memory it referred to. This pointer is the same pointer that was placed in storage by [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it."). If id does not belong to `type`, then NULL is returned.
The `id` parameter is the ID which is to be removed from its type.
The `type` parameter is the identifier for the ID type which `id` is supposed to belong to. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab). 

Note
    This function does NOT deallocate the memory that `id` refers to. The pointer returned by [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it.") must be deallocated by the user to avoid memory leaks. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga1f59a0d6e594ce7d61d85e1c16a0a381)H5Isearch()
void * H5Isearch  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  | [H5I_search_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a7400bdcded3eef3c4c4cf930173f9f72) |  _func_ ,   
|  | void * |  _key_ )  
Finds the memory referred to by an ID within the given ID type such that some criterion is satisfied.  

Parameters
     [in] | type | The identifier of the type to be searched   
---|---|---  
[in] | func | The function defining the search criteria   
[in] | key | A key for the search function 

Returns
    Returns a pointer to the object which satisfies the search function on success, NULL on failure.  
[H5Isearch()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga1f59a0d6e594ce7d61d85e1c16a0a381 "Finds the memory referred to by an ID within the given ID type such that some criterion is satisfied.") searches through a given ID type to find an object that satisfies the criteria defined by `func`. If such an object is found, the pointer to the memory containing this object is returned. Otherwise, NULL is returned. To do this, `func` is called on every member of type `type`. The first member to satisfy `func` is returned.
The `type` parameter is the identifier for the ID type which is to be searched. This identifier must have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab).
The parameter `func` is a function pointer to a function which takes three parameters. The first parameter is a `void*` and will be a pointer to the object to be tested. This is the same object that was placed in storage using [H5Iregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab90ab765fa23e8ffbafa228d3c8e82c5 "Registers an object under a type and returns an ID for it."). The second parameter is a [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) and is the ID of the object to be tested. The last parameter is a `void*`. This is the `key` parameter and can be used however the user finds helpful, or it can be ignored if it is not needed. `func` returns 0 if the object it is testing does not pass its criteria. A non-zero value should be returned if the object does pass its criteria. [H5I_search_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a7400bdcded3eef3c4c4cf930173f9f72) is defined in [H5Ipublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html) and is shown below. 
typedef int (*[H5I_search_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a7400bdcded3eef3c4c4cf930173f9f72))(void *obj, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, void *key);
The `key` parameter will be passed to the search function as a parameter. It can be used to further define the search at run-time. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gac137a599a2055909918e55955a69dfad)H5Itype_exists()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Itype_exists  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) | _type_ | ) |   
---|---|---|---|---|---  
Determines whether an identifier type is registered.  

Parameters
     [in] | type | Identifier type  
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5Itype_exists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gac137a599a2055909918e55955a69dfad "Determines whether an identifier type is registered.") determines whether the given identifier type, `type`, is registered with the library. 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


