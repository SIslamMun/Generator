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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title0 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title1 "H2")
  * [◆ H5A_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title2 "H2")
  * [◆ H5Acreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title3 "H2")
  * [◆ H5Aiterate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title4 "H2")
  * [◆ H5Dcreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title5 "H2")
  * [◆ H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title6 "H2")
  * [◆ H5Dopen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title7 "H2")
  * [◆ H5Dread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title8 "H2")
  * [◆ H5Dread_chunk_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title9 "H2")
  * [◆ H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title10 "H2")
  * [◆ H5E_auto_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title11 "H2")
  * [◆ H5E_error_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title12 "H2")
  * [◆ H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title13 "H2")
  * [◆ H5Eclear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title14 "H2")
  * [◆ H5Eclear_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title15 "H2")
  * [◆ H5Eget_auto_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title16 "H2")
  * [◆ H5Eprint_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title17 "H2")
  * [◆ H5Epush_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title18 "H2")
  * [◆ H5Eset_auto_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title19 "H2")
  * [◆ H5Ewalk_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title20 "H2")
  * [◆ H5F_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title21 "H2")
  * [◆ H5Fget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title22 "H2")
  * [◆ H5Gcreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title23 "H2")
  * [◆ H5Gopen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title24 "H2")
  * [◆ H5Iregister_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title25 "H2")
  * [◆ H5Iregister_type_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title26 "H2")
  * [◆ H5L_info_t [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title27 "H2")
  * [◆ H5L_info_t [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title28 "H2")
  * [◆ H5L_iterate_t [1/4]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title29 "H2")
  * [◆ H5L_iterate_t [2/4]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title30 "H2")
  * [◆ H5L_iterate_t [3/4]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title31 "H2")
  * [◆ H5L_iterate_t [4/4]](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title32 "H2")
  * [◆ H5Lget_info_by_idx_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title33 "H2")
  * [◆ H5Lget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title34 "H2")
  * [◆ H5Literate_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title35 "H2")
  * [◆ H5Literate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title36 "H2")
  * [◆ H5Lvisit_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title37 "H2")
  * [◆ H5Lvisit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title38 "H2")
  * [◆ H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title39 "H2")
  * [◆ H5O_info_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title40 "H2")
  * [◆ H5O_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title41 "H2")
  * [◆ H5O_iterate_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title42 "H2")
  * [◆ H5Oget_info_by_idx_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title43 "H2")
  * [◆ H5Oget_info_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title44 "H2")
  * [◆ H5Oget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title45 "H2")
  * [◆ H5Ovisit_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title46 "H2")
  * [◆ H5Ovisit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title47 "H2")
  * [◆ H5Pencode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title48 "H2")
  * [◆ H5Pencode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title49 "H2")
  * [◆ H5Pget_filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title50 "H2")
  * [◆ H5Pget_filter_by_id](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title51 "H2")
  * [◆ H5Pget_filter_by_id_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title52 "H2")
  * [◆ H5Pget_filter_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title53 "H2")
  * [◆ H5Pinsert](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title54 "H2")
  * [◆ H5Pinsert_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title55 "H2")
  * [◆ H5Pregister](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title56 "H2")
  * [◆ H5Pregister_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title57 "H2")
  * [◆ H5Rdereference_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title58 "H2")
  * [◆ H5Rget_obj_type_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title59 "H2")
  * [◆ H5Sencode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title60 "H2")
  * [◆ H5Tarray_create_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title61 "H2")
  * [◆ H5Tcommit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title62 "H2")
  * [◆ H5Tdecode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title63 "H2")
  * [◆ H5Tdecode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title64 "H2")
  * [◆ H5Tget_array_dims_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title65 "H2")
  * [◆ H5Topen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title66 "H2")
  * [◆ H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title67 "H2")
  * [◆ H5Z_class_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#title68 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#define-members)
H5version.h File Reference
##  Macros  
---  
#define  |  [H5A_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a04b85ea47893cb5538a75130533900a9) [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)  
#define  |  [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308)  
#define  |  [H5Acreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a21b7090dae115b324913a52ab085b057) 2  
#define  |  [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) [H5Aiterate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251)  
#define  |  [H5Aiterate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a0b54b3356921023fea6384154a5e4dc8) 2  
#define  |  [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)  
#define  |  [H5Dcreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2cb2264c54474285514e093ecc53812d) 2  
#define  |  [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)  
#define  |  [H5Dopen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#aced09b01bfb716412520c1f0a4defb99) 2  
#define  |  [H5Dread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e) [H5Dread_chunk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c)  
#define  |  [H5Dread_chunk_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a6ea12706de4846acf281e2056b5cea83) 2  
#define  |  [H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051) [H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca)  
#define  |  [H5E_auto_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ab899fd0758faf9aadacb758c44979428) 2  
#define  |  [H5E_error_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2592cae9c5fa61a6045b621cd94d3383) [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html)  
#define  |  [H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e) [H5E_walk2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aa0fc6445c613e4159a17d28ca61be825)  
#define  |  [H5Eclear](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac4c79bee16d0ffe8a6017bfdb66c9916) [H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2)  
#define  |  [H5Eclear_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3ca9a13c72ef7c2ef787d45e0cb023ef) 2  
#define  |  [H5Eget_auto](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa088620ef0b6c4ac2abcf57d61c8cdb8) [H5Eget_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga2eda33cbadd9be5bfddbaa91e863c936)  
#define  |  [H5Eget_auto_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2cb34a619050ebfbc90bfce8fde1c6b8) 2  
#define  |  [H5Eprint](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa7e93cb96b399e5853721258872435a8) [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56)  
#define  |  [H5Eprint_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac65ffe31a881f2610a8c94cb993506f4) 2  
#define  |  [H5Epush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4a7d9ca6b4f7bf521d29c85bbc5b7941) [H5Epush2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga17c2790837a1c1ea7e56b65d3c00a920)  
#define  |  [H5Epush_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3fea87cfbf23044dc6ee6991db0315b8) 2  
#define  |  [H5Eset_auto](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga245383d63fba5c4282ce8e9ef8488702) [H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)  
#define  |  [H5Eset_auto_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8344ab8b7666797e4400cb95ff961707) 2  
#define  |  [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0) [H5Ewalk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4ecc0f6a1ea5bb821373a5a7b8070655)  
#define  |  [H5Ewalk_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af7ae565c28408a89cf670889a4ec2fe7) 2  
#define  |  [H5F_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9994ec490861defacf6aed24d12a938a) [H5F_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__info2__t.html)  
#define  |  [H5Fget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae17036b3e36a8777328204e8bf073144) [H5Fget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e)  
#define  |  [H5Fget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#afe69be246e2747b73a72159a315a5189) 2  
#define  |  [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) [H5Gcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga86d93295965f750ef25dea2505a711d9)  
#define  |  [H5Gcreate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ae8ec3114e7970ceac40260041d66edbb) 2  
#define  |  [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245) [H5Gopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadab91e2dd7a7e253dcc0e4fe04b81403)  
#define  |  [H5Gopen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a76c98e4e3ad7bd3a5a480e59f446d20d) 2  
#define  |  [H5Iregister_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab) [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)  
#define  |  [H5Iregister_type_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af09c394c967f7c1fa3aee735921c50aa) 2  
#define  |  [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)  
#define  |  [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)  
#define  |  [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
#define  |  [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
#define  |  [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
#define  |  [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
#define  |  [H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)  
#define  |  [H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67) [H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)  
#define  |  [H5Lget_info_by_idx_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8c61d2d4817d8c3eaa6252043b490304) 2  
#define  |  [H5Lget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#adb82b5eee718b22652e14b18e466a62b) 2  
#define  |  [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) [H5Literate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73)  
#define  |  [H5Literate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga655b002428e0176c2fa23a0315fbbcc2) [H5Literate_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e)  
#define  |  [H5Literate_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a18218e95de866c426e9fdd9ac7f4a33d) 2  
#define  |  [H5Literate_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ae8d6974379891ba3f43709ccf5cbc7ac) 2  
#define  |  [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) [H5Lvisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66)  
#define  |  [H5Lvisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga138405315e233673741893e4e250f055) [H5Lvisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173)  
#define  |  [H5Lvisit_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a0a5cd62979043350f200d18e0b0c3da5) 2  
#define  |  [H5Lvisit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a13ff2bb0bfdbb7541ae2f567aaf15738) 2  
#define  |  [H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4) [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html)  
#define  |  [H5O_info_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a377c3c166e3937b657daf17b678cc361) 2  
#define  |  [H5O_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af397680764c0b285efb69404b9e421b9) [H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c)  
#define  |  [H5O_iterate_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8371f78a255da0e7a540c2486225658e) 2  
#define  |  [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) [H5Oget_info3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013)  
#define  |  [H5Oget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd) [H5Oget_info_by_idx3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa2f8884f7d3e7fd9b8549f5b59fd9eb)  
#define  |  [H5Oget_info_by_idx_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad39b7ae9e88bd342937a882a79243946) 3  
#define  |  [H5Oget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c) [H5Oget_info_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gabb69c962999e027cef0079bbb1282199)  
#define  |  [H5Oget_info_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#acee6aa6dd86a519b4fc9dceac94d2c17) 3  
#define  |  [H5Oget_info_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a30da0a37991573c68c2d5470798984b0) 3  
#define  |  [H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) [H5Ovisit3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a)  
#define  |  [H5Ovisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gab02a69e88b11404e7fd61f55344b186c) [H5Ovisit_by_name3](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga34815400b01df59c4dac19436124885a)  
#define  |  [H5Ovisit_by_name_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a63616f7a8a91c4d12007020eff85063b) 3  
#define  |  [H5Ovisit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad5768dea620b1bed1d6565bdf9244a3f) 3  
#define  |  [H5Pencode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8) [H5Pencode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa)  
#define  |  [H5Pencode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#aca058aeab1aba83130afce6febd0303e) 2  
#define  |  [H5Pget_filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15) [H5Pget_filter2](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga024d200a6a07e12f008a62c4e62d0bcc)  
#define  |  [H5Pget_filter_by_id](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac7aa336e7b1b9033cea2448ba623951f) [H5Pget_filter_by_id2](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga2d5e9df5f0e93abae11ee5edd82fcec3)  
#define  |  [H5Pget_filter_by_id_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af6c0d0080d28b0b3ddbb4e09bd3ddd0d) 2  
#define  |  [H5Pget_filter_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9913e361430b56e71f76d6d08c8e0d4d) 2  
#define  |  [H5Pinsert](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9ccdea50538c7cfde87a9fa63ea68555) [H5Pinsert2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga930e15d5f994e223bea80621ef3065d4)  
#define  |  [H5Pinsert_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a11c5bea3c2834bb92a2450ea0db136e8) 2  
#define  |  [H5Pregister](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a334ff323dfa6653ce21d0988ae7c73ba) [H5Pregister2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gaac3f957a5d3cbb4adc8b7ba2aa5f1719)  
#define  |  [H5Pregister_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a66ca006f0800bbcf6cdb37a87ce0143c) 2  
#define  |  [H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a) [H5Rdereference2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7)  
#define  |  [H5Rdereference_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dd741ca378477a48877122454539015) 2  
#define  |  [H5Rget_obj_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gafe413df448be0d230de922357fd7bc3b) [H5Rget_obj_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807)  
#define  |  [H5Rget_obj_type_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7a08f2870f6a271ad932e7568691c68f) 2  
#define  |  [H5Sencode](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga35e289baf490229e233296929fbf5208) [H5Sencode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga178ec7b8769ad5704a170d9bd3421074)  
#define  |  [H5Sencode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad7374945cce9e93c6f915d741befea51) 2  
#define  |  [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618)  
#define  |  [H5Tarray_create_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3c861ed10a0881a845dd1c73cd1fe887) 2  
#define  |  [H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) [H5Tcommit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga10352b6fa9ac58a7fbd5299496f1df31)  
#define  |  [H5Tcommit_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a1cead1b29370ba44a44aef35a1f89aa7) 2  
#define  |  [H5Tdecode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac90f7722dacad861c0a22507d3adf0dd) [H5Tdecode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga485114ecc0c366fda0d88833dd1083a1)  
#define  |  [H5Tdecode_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ab2feb62fde3b881036a004a1f7236119) 2  
#define  |  [H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) [H5Tget_array_dims2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769)  
#define  |  [H5Tget_array_dims_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac2ddd5b18787bef89be941488316d467) 2  
#define  |  [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) [H5Topen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga7e65e77634f1fb4ba38cbcdab9a59bc2)  
#define  |  [H5Topen_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a1429683442094f9484fca6278c0d2f66) 2  
#define  |  [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html)  
#define  |  [H5Z_class_t_vers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a29e27de6e11a401bed2d36715de06167) 2  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a04b85ea47893cb5538a75130533900a9)H5A_operator_t
#define H5A_operator_t [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a21b7090dae115b324913a52ab085b057)H5Acreate_vers
#define H5Acreate_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a0b54b3356921023fea6384154a5e4dc8)H5Aiterate_vers
#define H5Aiterate_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2cb2264c54474285514e093ecc53812d)H5Dcreate_vers
#define H5Dcreate_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)H5Dopen
#define H5Dopen [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#aced09b01bfb716412520c1f0a4defb99)H5Dopen_vers
#define H5Dopen_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e)H5Dread_chunk
#define H5Dread_chunk [H5Dread_chunk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a6ea12706de4846acf281e2056b5cea83)H5Dread_chunk_vers
#define H5Dread_chunk_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051)H5E_auto_t
#define H5E_auto_t [H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ab899fd0758faf9aadacb758c44979428)H5E_auto_t_vers
#define H5E_auto_t_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2592cae9c5fa61a6045b621cd94d3383)H5E_error_t
#define H5E_error_t [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e)H5E_walk_t
#define H5E_walk_t [H5E_walk2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aa0fc6445c613e4159a17d28ca61be825)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac4c79bee16d0ffe8a6017bfdb66c9916)H5Eclear
#define H5Eclear [H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3ca9a13c72ef7c2ef787d45e0cb023ef)H5Eclear_vers
#define H5Eclear_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2cb34a619050ebfbc90bfce8fde1c6b8)H5Eget_auto_vers
#define H5Eget_auto_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac65ffe31a881f2610a8c94cb993506f4)H5Eprint_vers
#define H5Eprint_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3fea87cfbf23044dc6ee6991db0315b8)H5Epush_vers
#define H5Epush_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8344ab8b7666797e4400cb95ff961707)H5Eset_auto_vers
#define H5Eset_auto_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af7ae565c28408a89cf670889a4ec2fe7)H5Ewalk_vers
#define H5Ewalk_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9994ec490861defacf6aed24d12a938a)H5F_info_t
#define H5F_info_t [H5F_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__info2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#afe69be246e2747b73a72159a315a5189)H5Fget_info_vers
#define H5Fget_info_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ae8ec3114e7970ceac40260041d66edbb)H5Gcreate_vers
#define H5Gcreate_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a76c98e4e3ad7bd3a5a480e59f446d20d)H5Gopen_vers
#define H5Gopen_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab)H5Iregister_type
#define H5Iregister_type [H5Iregister_type2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af09c394c967f7c1fa3aee735921c50aa)H5Iregister_type_vers
#define H5Iregister_type_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4)H5L_info_t [1/2]
#define H5L_info_t [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4)H5L_info_t [2/2]
#define H5L_info_t [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407)H5L_iterate_t [1/4]
#define H5L_iterate_t [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407)H5L_iterate_t [2/4]
#define H5L_iterate_t [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407)H5L_iterate_t [3/4]
#define H5L_iterate_t [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407)H5L_iterate_t [4/4]
#define H5L_iterate_t [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8c61d2d4817d8c3eaa6252043b490304)H5Lget_info_by_idx_vers
#define H5Lget_info_by_idx_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#adb82b5eee718b22652e14b18e466a62b)H5Lget_info_vers
#define H5Lget_info_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a18218e95de866c426e9fdd9ac7f4a33d)H5Literate_by_name_vers
#define H5Literate_by_name_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ae8d6974379891ba3f43709ccf5cbc7ac)H5Literate_vers
#define H5Literate_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a0a5cd62979043350f200d18e0b0c3da5)H5Lvisit_by_name_vers
#define H5Lvisit_by_name_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a13ff2bb0bfdbb7541ae2f567aaf15738)H5Lvisit_vers
#define H5Lvisit_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4)H5O_info_t
#define H5O_info_t [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a377c3c166e3937b657daf17b678cc361)H5O_info_t_vers
#define H5O_info_t_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af397680764c0b285efb69404b9e421b9)H5O_iterate_t
#define H5O_iterate_t [H5O_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a564cec62aef0389091ad21d235aa321c)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8371f78a255da0e7a540c2486225658e)H5O_iterate_t_vers
#define H5O_iterate_t_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad39b7ae9e88bd342937a882a79243946)H5Oget_info_by_idx_vers
#define H5Oget_info_by_idx_vers 3  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#acee6aa6dd86a519b4fc9dceac94d2c17)H5Oget_info_by_name_vers
#define H5Oget_info_by_name_vers 3  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a30da0a37991573c68c2d5470798984b0)H5Oget_info_vers
#define H5Oget_info_vers 3  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a63616f7a8a91c4d12007020eff85063b)H5Ovisit_by_name_vers
#define H5Ovisit_by_name_vers 3  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad5768dea620b1bed1d6565bdf9244a3f)H5Ovisit_vers
#define H5Ovisit_vers 3  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8)H5Pencode
#define H5Pencode [H5Pencode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#aca058aeab1aba83130afce6febd0303e)H5Pencode_vers
#define H5Pencode_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15)H5Pget_filter
#define H5Pget_filter [H5Pget_filter2](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga024d200a6a07e12f008a62c4e62d0bcc)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac7aa336e7b1b9033cea2448ba623951f)H5Pget_filter_by_id
#define H5Pget_filter_by_id [H5Pget_filter_by_id2](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga2d5e9df5f0e93abae11ee5edd82fcec3)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af6c0d0080d28b0b3ddbb4e09bd3ddd0d)H5Pget_filter_by_id_vers
#define H5Pget_filter_by_id_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9913e361430b56e71f76d6d08c8e0d4d)H5Pget_filter_vers
#define H5Pget_filter_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9ccdea50538c7cfde87a9fa63ea68555)H5Pinsert
#define H5Pinsert [H5Pinsert2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga930e15d5f994e223bea80621ef3065d4)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a11c5bea3c2834bb92a2450ea0db136e8)H5Pinsert_vers
#define H5Pinsert_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a334ff323dfa6653ce21d0988ae7c73ba)H5Pregister
#define H5Pregister [H5Pregister2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gaac3f957a5d3cbb4adc8b7ba2aa5f1719)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a66ca006f0800bbcf6cdb37a87ce0143c)H5Pregister_vers
#define H5Pregister_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dd741ca378477a48877122454539015)H5Rdereference_vers
#define H5Rdereference_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7a08f2870f6a271ad932e7568691c68f)H5Rget_obj_type_vers
#define H5Rget_obj_type_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ad7374945cce9e93c6f915d741befea51)H5Sencode_vers
#define H5Sencode_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3c861ed10a0881a845dd1c73cd1fe887)H5Tarray_create_vers
#define H5Tarray_create_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a1cead1b29370ba44a44aef35a1f89aa7)H5Tcommit_vers
#define H5Tcommit_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac90f7722dacad861c0a22507d3adf0dd)H5Tdecode
#define H5Tdecode [H5Tdecode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga485114ecc0c366fda0d88833dd1083a1)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ab2feb62fde3b881036a004a1f7236119)H5Tdecode_vers
#define H5Tdecode_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac2ddd5b18787bef89be941488316d467)H5Tget_array_dims_vers
#define H5Tget_array_dims_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a1429683442094f9484fca6278c0d2f66)H5Topen_vers
#define H5Topen_vers 2  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba)H5Z_class_t
#define H5Z_class_t [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a29e27de6e11a401bed2d36715de06167)H5Z_class_t_vers
#define H5Z_class_t_vers 2  
---  
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5version.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


