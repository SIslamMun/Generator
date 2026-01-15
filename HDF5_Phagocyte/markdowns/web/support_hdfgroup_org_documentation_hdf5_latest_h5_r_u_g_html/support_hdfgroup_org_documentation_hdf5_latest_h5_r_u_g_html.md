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
  * [ HDF5 References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title1 "H2")
  * [ Deprecated API](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title2 "H2")
  * [ New API](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title3 "H2")
  * [ API Compatibility](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title4 "H2")
  * [ Usage Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 References
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 References
HDF5 references allow users to reference existing HDF5 objects (file, group, dataset, named datatype, or attribute) as well as selections within datasets.
The original API, now deprecated, was extended in order to add the ability to reference attributes as well as objects in external files. Additionally, there were some inherent limitations within the older API that restricted its use with virtual object layer (VOL) connectors, which do not necessarily follow HDF5’s native file format.
The newer API introduced a single opaque reference type, which not only has the advantage of hiding the internal representation of references, but it also allows for future extensions to be added more seamlessly.
##  Introduction
The deprecated HDF5 reference API only allowed users to create references to HDF5 objects (groups, datasets) and regions within a dataset. There were some limitations: it defined two separate reference types [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081) and [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html); the former directly mapped to an [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) type that did not allow for external references, while the latter mapped to an HDF5 global heap entry, which was specific to native HDF5 and was created and written to the file when the reference was created. This prevented users from creating region references when the file is opened read-only, it was also not suitable for use outside of native HDF5 files. The newer API addressed these limitations by introducing a single abstract [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) type as well as missing reference types such as attribute references and external references (i.e., references to objects in an external file).
##  Deprecated API
There is no support for attribute references; references are only valid within the container that they reference; the size of the reference types are tied to the definition of an [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) or an entry in the file’s global heap, which only exists in native HDF5.
###  Limitations
  * The [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d "Creates a reference.") signature forces users to constantly pass ([H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) as a space_id, in the case where the reference type is not a region reference. 
  * The size of region references was defined as the size required to encode a global heap ID, this definition forces references to be written to the file at the time of their creation, hence preventing them to be created from a file that is opened read-only (e.g, when creating references to a file that one does not want to/cannot modify).


###  Deprecated Methods
The original API before [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html) 1.12.0 is defined below: 
// Deprecated reference buffer sizes that are kept for backward compatibility
#define H5R_OBJ_REF_BUF_SIZE sizeof(haddr_t)
#define H5R_DSET_REG_REF_BUF_SIZE (sizeof(haddr_t) + 4)
// Reference types
typedef enum [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) {
[H5R_BADTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea4fc6b2cf09334d77851f8b208105d58f) = (-1), // Invalid Reference Type
[H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), // Object reference
[H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), // Dataset Region Reference
[H5R_MAXTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eaa89d43156d194e6538203cf7f4b6bda0) // Highest type (Invalid as true type)
} [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e);
// Object reference structure for user's code
// This needs to be large enough to store largest haddr_t on a worst case
// machine (8 bytes currently).
typedef [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081);
// Dataset Region reference structure for user's code
// (Buffer to store heap ID and index)
// This needs to be large enough to store largest haddr_t in a worst case
// machine (8 bytes currently) plus an int
typedef unsigned char [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html)[[H5R_DSET_REG_REF_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af6fd702307b37ca58c21e2ca1c771415)];
// Prototypes
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(void *ref, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Rdereference2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oapl_id, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type, const void *ref);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dataset, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type, const void *ref);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rget_obj_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type, const void *_ref, [H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) *obj_type);
ssize_t [H5Rget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga4c112c388f697324270fd085100dccaa)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type, const void *ref, char *name , size_t size);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec)
H5O_type_t
**Definition** H5Opublic.h:107
[H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0)
#define H5R_DATASET_REGION
**Definition** H5Rpublic.h:625
[H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e)
H5R_type_t
**Definition** H5Rpublic.h:51
[H5R_BADTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea4fc6b2cf09334d77851f8b208105d58f)
@ H5R_BADTYPE
**Definition** H5Rpublic.h:52
[H5R_MAXTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eaa89d43156d194e6538203cf7f4b6bda0)
@ H5R_MAXTYPE
**Definition** H5Rpublic.h:58
[H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2)
#define H5R_OBJECT
**Definition** H5Rpublic.h:624
[hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081)
haddr_t hobj_ref_t
**Definition** H5Rpublic.h:71
[H5R_DSET_REG_REF_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af6fd702307b37ca58c21e2ca1c771415)
#define H5R_DSET_REG_REF_BUF_SIZE
**Definition** H5Rpublic.h:30
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)
herr_t H5Rcreate(void *ref, hid_t loc_id, const char *name, H5R_type_t ref_type, hid_t space_id)
Creates a reference.
[H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f)
hid_t H5Rget_region(hid_t dataset, H5R_type_t ref_type, const void *ref)
Sets up a dataspace and selection as specified by a region reference.
[H5Rget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga4c112c388f697324270fd085100dccaa)
ssize_t H5Rget_name(hid_t loc_id, H5R_type_t ref_type, const void *ref, char *name, size_t size)
Retrieves a name for a referenced object.
[H5Rget_obj_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807)
herr_t H5Rget_obj_type2(hid_t id, H5R_type_t ref_type, const void *ref, H5O_type_t *obj_type)
Retrieves the type of object that an object reference points to.
[H5Rdereference2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7)
hid_t H5Rdereference2(hid_t obj_id, hid_t oapl_id, H5R_type_t ref_type, const void *ref)
Opens the HDF5 object referenced.
[hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html)
**Definition** H5Rpublic.h:86
##  New API
The current API is defined below: 
// Deprecated reference buffer sizes that are kept for backward compatibility
#define H5R_OBJ_REF_BUF_SIZE sizeof(haddr_t)
#define H5R_DSET_REG_REF_BUF_SIZE (sizeof(haddr_t) + 4)
// Default reference buffer size.
#define H5R_REF_BUF_SIZE (64)
// Reference types allowed.
typedef enum {
[H5R_BADTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea4fc6b2cf09334d77851f8b208105d58f) = (-1), // Invalid reference type
[H5R_OBJECT1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eae218034c065fc240fab1b2c709b7e23c) = 0, // Backward compatibility (object)
[H5R_DATASET_REGION1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eac9d8cfad097bec7d118333303f67c916) = 1, // Backward compatibility (region)
[H5R_OBJECT2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea58376ddd57d594982e0034a3ca2e5bad) = 2, // Object reference
[H5R_DATASET_REGION2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea2099d2dee8001ab3ce2e7041af5c93d6) = 3, // Region reference
[H5R_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eae63b3cfa3688767f46335fad6db981a5) = 4, // Attribute Reference
[H5R_MAXTYPE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eaa89d43156d194e6538203cf7f4b6bda0) = 5 // Highest type (invalid)
} [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e);
// Deprecated object reference type that is used with deprecated reference APIs.
// This type can only be used with the "native" HDF5 VOL connector.
typedef [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081);
// Deprecated dataset region reference type that is used with deprecated reference APIs.
// This type can only be used with the "native" HDF5 VOL connector.
typedef struct {
uint8_t __data[[H5R_DSET_REG_REF_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af6fd702307b37ca58c21e2ca1c771415)];
} [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html);
// Opaque reference type. The same reference type is used for object,
// dataset region and attribute references. This is the type that
// should always be used with the current reference API.
typedef struct {
union {
uint8_t __data[[H5R_REF_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a30e65b87f330f177dc5ae434104af580)]; // opaque data
int64_t align; // ensures alignment
} u;
} [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html);
// Constructors
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rcreate_attr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gacae1bf2263befeac54d81f27995721f8)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, const char *attr_name, [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)([H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr);
// Info
[H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) [H5Rget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga56989d33835ec289269d366dc28aa4ad)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr);
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5Requal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga8d7dac2c604356f2fc657f36a201aea0)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref1_ptr, const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref2_ptr);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad5b7b117cfaea90d1e4786304c58c55c)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *src_ref_ptr, [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *dst_ref_ptr);
// Dereference
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oapl_id);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Ropen_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oapl_id);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Ropen_attr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga2a00d73b9a13b9069d0ad39225dd9f71)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id);
// Get type
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Rget_obj_type3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga6fb098a1a4c9369d76dbeeaf40d6d2af)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) *obj_type);
// Get name
ssize_t [H5Rget_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga659d950e37122a59d50612b407a48900)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, char *name, size_t size);
ssize_t [H5Rget_obj_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gac5a0a3a874cd3c4dc630579521539bb6)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, char *name, size_t size);
ssize_t [H5Rget_attr_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaf9c6823346cc34f02b9f5d11e56344a0)(const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, char *name, size_t size);
[H5R_REF_BUF_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a30e65b87f330f177dc5ae434104af580)
#define H5R_REF_BUF_SIZE
**Definition** H5Rpublic.h:38
[H5R_DATASET_REGION2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea2099d2dee8001ab3ce2e7041af5c93d6)
@ H5R_DATASET_REGION2
**Definition** H5Rpublic.h:56
[H5R_OBJECT2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea58376ddd57d594982e0034a3ca2e5bad)
@ H5R_OBJECT2
**Definition** H5Rpublic.h:55
[H5R_DATASET_REGION1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eac9d8cfad097bec7d118333303f67c916)
@ H5R_DATASET_REGION1
**Definition** H5Rpublic.h:54
[H5R_OBJECT1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eae218034c065fc240fab1b2c709b7e23c)
@ H5R_OBJECT1
**Definition** H5Rpublic.h:53
[H5R_ATTR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eae63b3cfa3688767f46335fad6db981a5)
@ H5R_ATTR
**Definition** H5Rpublic.h:57
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)
int htri_t
**Definition** H5public.h:282
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)
herr_t H5Rdestroy(H5R_ref_t *ref_ptr)
Closes a reference.
[H5Ropen_attr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga2a00d73b9a13b9069d0ad39225dd9f71)
hid_t H5Ropen_attr(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t aapl_id)
Opens the HDF5 attribute referenced.
[H5Ropen_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a)
hid_t H5Ropen_region(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t oapl_id)
Sets up a dataspace and selection as specified by a region reference.
[H5Rget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga56989d33835ec289269d366dc28aa4ad)
H5R_type_t H5Rget_type(const H5R_ref_t *ref_ptr)
Retrieves the type of a reference.
[H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)
herr_t H5Rcreate_region(hid_t loc_id, const char *name, hid_t space_id, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates a region reference.
[H5Rget_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga659d950e37122a59d50612b407a48900)
ssize_t H5Rget_file_name(const H5R_ref_t *ref_ptr, char *name, size_t size)
Retrieves the file name for a referenced object.
[H5Rget_obj_type3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga6fb098a1a4c9369d76dbeeaf40d6d2af)
herr_t H5Rget_obj_type3(H5R_ref_t *ref_ptr, hid_t rapl_id, H5O_type_t *obj_type)
Retrieves the type of object that an object reference points to.
[H5Requal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga8d7dac2c604356f2fc657f36a201aea0)
htri_t H5Requal(const H5R_ref_t *ref1_ptr, const H5R_ref_t *ref2_ptr)
Determines whether two references are equal.
[H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)
hid_t H5Ropen_object(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t oapl_id)
Opens the HDF5 object referenced.
[H5Rget_obj_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gac5a0a3a874cd3c4dc630579521539bb6)
ssize_t H5Rget_obj_name(H5R_ref_t *ref_ptr, hid_t rapl_id, char *name, size_t size)
Retrieves the object name for a referenced object.
[H5Rcreate_attr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gacae1bf2263befeac54d81f27995721f8)
herr_t H5Rcreate_attr(hid_t loc_id, const char *name, const char *attr_name, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates an attribute reference.
[H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)
herr_t H5Rcreate_object(hid_t loc_id, const char *name, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates an object reference.
[H5Rcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad5b7b117cfaea90d1e4786304c58c55c)
herr_t H5Rcopy(const H5R_ref_t *src_ref_ptr, H5R_ref_t *dst_ref_ptr)
Copies an existing reference.
[H5Rget_attr_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaf9c6823346cc34f02b9f5d11e56344a0)
ssize_t H5Rget_attr_name(const H5R_ref_t *ref_ptr, char *name, size_t size)
Retrieves the attribute name for a referenced object.
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html)
**Definition** H5Rpublic.h:97
References can be stored and retrieved from a file by invoking the [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") and [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") functions with this single predefined type: [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc).
The advantage of a single type is that it becomes easier for users to mix references of different types. It is also more in line with the opaque type now defined for references. Note that when reading references back from a file, the library may, in consequence of this new design, allocate memory for each of these references. To release the memory, one must either call [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095 "Closes a reference.") on each of the references or, for convenience, call the new [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.") function on the buffer that contains the array of references (type can be compound type, array).
As mentioned, instead of having separate routines for both vlen and reference types, we unify the existing: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Dvlen_reclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf);
[H5Dvlen_reclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119)
herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id, hid_t dxpl_id, void *buf)
Reclaims variable-length (VL) datatype memory buffers.
to 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf);
[H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)
herr_t H5Treclaim(hid_t type_id, hid_t space_id, hid_t plist_id, void *buf)
Reclaims the variable length (VL) datatype memory buffers.
##  API Compatibility
To preserve compatibility with applications and middleware libraries that have been using the existing reference API, we keep the existing [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d "Creates a reference."), [H5Rdereference2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7 "Opens the HDF5 object referenced."), [H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f "Sets up a dataspace and selection as specified by a region reference."), [H5Rget_obj_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807 "Retrieves the type of object that an object reference points to.") and [H5Rget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga4c112c388f697324270fd085100dccaa "Retrieves a name for a referenced object.") routines, but moved to the deprecated API list of functions.
It is important to note though that these routines only support the original reference types, noted as [H5R_OBJECT1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eae218034c065fc240fab1b2c709b7e23c) and [H5R_DATASET_REGION1](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056eac9d8cfad097bec7d118333303f67c916) respectively. Any other reference type passed to these routines will return an error. For convenience and compatibility with previous versions of the library we define both [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2) and [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0) to map to the original reference types 
// Versions for compatibility
#define H5R_OBJECT H5R_OBJECT1
#define H5R_DATASET_REGION H5R_DATASET_REGION1
When creating and accessing references through these deprecated routines, users are still expected to use the datatypes which describe the [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081) and [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) types, [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1) and [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e).
One important aspect of these changes is to ensure that previously written data can still be readable after those revisions and that new files produced will not create any undefined behavior when used with previous versions of the library. Backward as well as forward compatibility is summarized in the table:
Version | Old File Format/Old API | Old File Format/New API | New File Format/Old API | New File Format/New API   
---|---|---|---|---  
< 1.12.0 | No change | N/A | Datatype version bump prevents from reading unknown reference types | N/A   
≥ 1.12.0  | Read and write references through old datatypes and use [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081) and [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) types  | Read and write using [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) to convert to new [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) type  | Cannot use old API with new reference types  | Can use opaque [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) type for all reference types   
Because previous library versions do not have a way of detecting when new unknown references types are read, we have to increment the global version of the datatypes, so that early detection can be done and the appropriate error is returned to the user. For versions prior to this change, the library will return an error when the datatype encountered has a version number greater than the currently supported version. Also, to prevent datatype version changes in the future, all library branches are now patched to check for unknown reference types.
When reading old data with the new library version, one can either keep using the [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1) and [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e) datatypes, which can be queried when opening a dataset, for example using [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset."), or use the [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) datatype, which will trigger automatic type conversion. The [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1) and [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e) datatypes require the use of the respective [hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081) and [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) types, which can only be used with the old API functions. These types do not embed all the required information to be simply cast to an [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) type. When an [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) type is desired, the [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) datatype must be used, allowing old reference data to be used with the new API.
##  Usage Examples
###  External References
The example below illustrates the use of the new API with files that are opened read-only. Created references to the objects in that file are stored into a separate file, and accessed from that file, without the user explicitly opening the original file that was referenced. 
#include <stdlib.h>
#include "hdf5.h"
#include <assert.h>
#define H5FILE_NAME1 "refer_extern1.h5"
#define H5FILE_NAME2 "refer_extern2.h5"
#define NDIMS 1 // Number of dimensions
#define BUF_SIZE 4 // Size of example buffer
#define NREFS 1 // Number of references
int main(void) {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file1, dset1, space1;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dset1_dims[NDIMS] = { BUF_SIZE };
int dset_buf[BUF_SIZE];
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file2, dset2, space2;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dset2_dims[NDIMS] = { NREFS };
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) ref_buf[NREFS] = { 0 };
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) obj_type;
int i;
for (i = 0; i < BUF_SIZE; i++)
dset_buf[i] = i;
// Create file with one dataset and close it
file1 = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(H5FILE_NAME1, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
space1 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(NDIMS, dset1_dims, NULL);
dset1 = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file1, "dataset1", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), space1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset1, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset1);
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(space1);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file1);
// Create reference to dataset1 in "refer_extern1.h5"
file1 = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(H5FILE_NAME1, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)(file1, "dataset1", &ref_buf[0]);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file1);
// Store reference in dataset in separate file "refer_extern2.h5"
file2 = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(H5FILE_NAME2, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
space2 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(NDIMS, dset2_dims, NULL);
dset2 = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file2, "references", [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), space2, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset2, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset2);
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(space2);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file2);
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref_buf[0]);
// Read reference back from "refer_extern2.h5"
file2 = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(H5FILE_NAME2, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dset2 = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file2, "references", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset2, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset2);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file2);
// Access reference and read dataset data without opening original file
assert([H5Rget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga56989d33835ec289269d366dc28aa4ad)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&ref_buf[0]) == [H5R_OBJECT2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea58376ddd57d594982e0034a3ca2e5bad));
[H5Rget_obj_type3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga6fb098a1a4c9369d76dbeeaf40d6d2af)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&ref_buf[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &obj_type);
assert(obj_type == [H5O_TYPE_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdeca5ca744a77f8cd2b28dda90c37807ae31));
dset1 = [H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&ref_buf[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset1, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset1);
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref_buf[0]);
for (i = 0; i < BUF_SIZE; i++)
assert(dset_buf[i] == i);
return 0;
}
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394)
#define H5F_ACC_RDONLY
**Definition** H5Fpublic.h:28
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5O_TYPE_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdeca5ca744a77f8cd2b28dda90c37807ae31)
@ H5O_TYPE_DATASET
**Definition** H5Opublic.h:110
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)
hid_t H5Dopen2(hid_t loc_id, const char *name, hid_t dapl_id)
Opens an existing dataset.
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
Creates a new dataset and links it into the file.
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)
herr_t H5Sclose(hid_t space_id)
Releases and terminates access to a dataspace.
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc)
#define H5T_STD_REF
**Definition** H5Tpublic.h:576
###  Backward Compatibility and New API
The example below illustrates the use of the new API with a file that was written using the old-style reference API, showing how one can take advantage of the automatic type conversion from old reference type to new reference type. 
#include <stdlib.h>
#include "hdf5.h"
#include <assert.h>
#define H5FILE_NAME "refer_deprec.h5"
#define NDIMS 1 // Number of dimensions
#define BUF_SIZE 4 // Size of example buffer
#define NREFS 1 // Number of references
int main(void) {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file1, dset1, space1;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dset1_dims[NDIMS] = { BUF_SIZE };
int dset_buf[BUF_SIZE];
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset2, space2;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dset2_dims[NDIMS] = { NREFS };
[hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081) ref_buf[NREFS] = { 0 };
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) new_ref_buf[NREFS] = { 0 };
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) obj_type;
int i;
for (i = 0; i < BUF_SIZE; i++)
dset_buf[i] = i;
// Create file with one dataset and close it
file1 = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(H5FILE_NAME, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
space1 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(NDIMS, dset1_dims, NULL);
dset1 = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file1, "dataset1", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), space1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset1, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset1);
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(space1);
// Create reference to dataset1 with deprecated API
// (reminder: there is no destroy call for those references)
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&ref_buf[0], file1, "dataset1", [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba));
// Store reference in separate dataset using deprecated reference type
space2 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(NDIMS, dset2_dims, NULL);
dset2 = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file1, "references", [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1), space2, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset2, [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref_buf); [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset2);
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(space2);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file1);
// Read reference from file using new reference type
file1 = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(H5FILE_NAME, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dset2 = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file1, "references", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset2, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), new_ref_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset2);
// Access reference and read dataset data through new API
assert([H5Rget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga56989d33835ec289269d366dc28aa4ad)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&new_ref_buf[0]) == [H5R_OBJECT2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056ea58376ddd57d594982e0034a3ca2e5bad));
[H5Rget_obj_type3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga6fb098a1a4c9369d76dbeeaf40d6d2af)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&new_ref_buf[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &obj_type);
assert(obj_type == [H5O_TYPE_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdeca5ca744a77f8cd2b28dda90c37807ae31));
dset1 = [H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)((const [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)&new_ref_buf[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset1, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_buf);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset1);
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&new_ref_buf[0]);
for (i = 0; i < BUF_SIZE; i++)
assert(dset_buf[i] == i);
return 0;
}
[H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)
#define H5I_INVALID_HID
**Definition** H5Ipublic.h:71
[H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1)
#define H5T_STD_REF_OBJ
**Definition** H5Tpublic.h:566
Previous Chapter [HDF5 Map Object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#sec_map) - Next Chapter [HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#sec_filter_plugins)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


