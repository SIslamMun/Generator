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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title3 "H2")
  * [◆ H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title4 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title5 "H2")
  * [◆ H5Dchunk_iter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title6 "H2")
  * [◆ H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title7 "H2")
  * [◆ H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title8 "H2")
  * [◆ H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title9 "H2")
  * [◆ H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title10 "H2")
  * [◆ H5Dextend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title11 "H2")
  * [◆ H5Dfill()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title12 "H2")
  * [◆ H5Dflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title13 "H2")
  * [◆ H5Dgather()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title14 "H2")
  * [◆ H5Dget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title15 "H2")
  * [◆ H5Dget_chunk_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title16 "H2")
  * [◆ H5Dget_chunk_info_by_coord()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title17 "H2")
  * [◆ H5Dget_chunk_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title18 "H2")
  * [◆ H5Dget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title19 "H2")
  * [◆ H5Dget_num_chunks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title20 "H2")
  * [◆ H5Dget_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title21 "H2")
  * [◆ H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title22 "H2")
  * [◆ H5Dget_space_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title23 "H2")
  * [◆ H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title24 "H2")
  * [◆ H5Dget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title25 "H2")
  * [◆ H5Diterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title26 "H2")
  * [◆ H5Dopen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title27 "H2")
  * [◆ H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title28 "H2")
  * [◆ H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title29 "H2")
  * [◆ H5Dread_chunk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title30 "H2")
  * [◆ H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title31 "H2")
  * [◆ H5Dread_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title32 "H2")
  * [◆ H5Drefresh()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title33 "H2")
  * [◆ H5Dscatter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title34 "H2")
  * [◆ H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title35 "H2")
  * [◆ H5Dvlen_get_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title36 "H2")
  * [◆ H5Dvlen_reclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title37 "H2")
  * [◆ H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title38 "H2")
  * [◆ H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title39 "H2")
  * [◆ H5Dwrite_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#title40 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#func-members)
Datasets (H5D)
## Detailed Description
Use the functions in this module to manage HDF5 datasets, including the transfer of data between memory and disk and the description of dataset properties. Datasets are used by other HDF5 APIs and referenced either by name or by a handle. Such handles can be obtained by either creating or opening the dataset.
Typical stages in the HDF5 dataset life cycle are shown below in introductory examples.
Create | Read   
---|---  
94 { 95 __label__ fail_lcpl, fail_dset, fail_file; 96 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, lcpl, fspace, dset; 97 98 unsigned mode = [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a); 99 char file_name[] = "d1.h5"; 100 // link names can be arbitrary Unicode strings 101 char dset_name[] = "σύνολο/δεδομένων"; 102 103 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 104 ret_val = EXIT_FAILURE; 105 goto fail_file; 106 } 107 if ((lcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 108 ret_val = EXIT_FAILURE; 109 goto fail_lcpl; 110 } 111 // use UTF-8 encoding for link names 112 if ([H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)(lcpl, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)) < 0) { 113 ret_val = EXIT_FAILURE; 114 goto fail_fspace; 115 } 116 // create intermediate groups as needed 117 if ([H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c)(lcpl, 1) < 0) { 118 ret_val = EXIT_FAILURE; 119 goto fail_fspace; 120 } 121 // create a 1D dataspace 122 if ((fspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){10}, NULL)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 123 ret_val = EXIT_FAILURE; 124 goto fail_fspace; 125 } 126 // create a 32-bit integer dataset 127 if ((dset = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file, dset_name, [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), fspace, lcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == 128 [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 129 ret_val = EXIT_FAILURE; 130 goto fail_dset; 131 } 132 133 [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset); 134fail_dset: 135 [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace); 136fail_fspace: 137 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(lcpl); 138fail_lcpl: 139 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file); 140fail_file:; 141 } |  145 { 146 __label__ fail_dset, fail_file; 147 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset; 148 149 unsigned mode = [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394); 150 char file_name[] = "d1.h5"; 151 // assume a priori knowledge of dataset name and size 152 char dset_name[] = "σύνολο/δεδομένων"; 153 int elts[10]; 154 155 if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 156 ret_val = EXIT_FAILURE; 157 goto fail_file; 158 } 159 if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 160 ret_val = EXIT_FAILURE; 161 goto fail_dset; 162 } 163 // read all dataset elements 164 if ([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), elts) < 0) 165 ret_val = EXIT_FAILURE; 166 167 // do something w/ the dataset elements 168 169 [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset); 170fail_dset: 171 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file); 172fail_file:; 173 }  
Update | Delete   
177 { 178 __label__ fail_update, fail_fspace, fail_dset, fail_file; 179 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset, fspace; 180 181 unsigned mode = [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4); 182 char file_name[] = "d1.h5"; 183 char dset_name[] = "σύνολο/δεδομένων"; 184 int new_elts[6][2] = {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}}; 185 186 if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 187 ret_val = EXIT_FAILURE; 188 goto fail_file; 189 } 190 if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 191 ret_val = EXIT_FAILURE; 192 goto fail_dset; 193 } 194 // get the dataset's dataspace 195 if ((fspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dset)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 196 ret_val = EXIT_FAILURE; 197 goto fail_fspace; 198 } 199 // select the first 5 elements in odd positions 200 if ([H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){1}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){2}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){5}, 201 NULL) < 0) { 202 ret_val = EXIT_FAILURE; 203 goto fail_update; 204 } 205 206 // (implicitly) select and write the first 5 elements of the second column of NEW_ELTS 207 if ([H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), fspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), new_elts) < 0) 208 ret_val = EXIT_FAILURE; 209 210fail_update: 211 [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace); 212fail_fspace: 213 [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset); 214fail_dset: 215 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file); 216fail_file:; 217 } |  221 { 222 __label__ fail_delete, fail_file; 223 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file; 224 225 unsigned mode = [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4); 226 char file_name[] = "d1.h5"; 227 char group_name[] = "σύνολο"; 228 char dset_name[] = "σύνολο/δεδομένων"; 229 230 if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 231 ret_val = EXIT_FAILURE; 232 goto fail_file; 233 } 234 // delete (unlink) the dataset 235 if ([H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)) < 0) { 236 ret_val = EXIT_FAILURE; 237 goto fail_delete; 238 } 239 // the previous call deletes (unlinks) only the dataset 240 if ([H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf)(file, group_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)) < 0) { 241 ret_val = EXIT_FAILURE; 242 goto fail_delete; 243 } 244 245fail_delete: 246 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file); 247fail_file:; 248 }  
##  Macros  
---  
#define  |  [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dchunk_iter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac482c2386aa3aea4c44730a627a7adb8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5D_chunk_iter_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a7a2008481c5cef463cbd0943a7042609) cb, void *op_data)  
| Iterate over all chunks of a chunked dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Closes the specified dataset.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dcreate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id)  
| Creates a dataset at the specified location.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id)  
| Creates a new dataset and links it into the file.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id)  
| Creates a dataset in a file without linking it into the file structure.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size[])  
| Extends a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dfill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8d4a57e2b2b8c95cfecf6f75bdaa8343) (const void *fill, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fill_type_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) buf_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id)  
| Fills dataspace elements with a fill value in a memory buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Flushes all buffers associated with a dataset to disk.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dgather](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_space_id, const void *src_buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, size_t dst_buf_size, void *dst_buf, [H5D_gather_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#ab0c47a79dc28343e9c1288e911ae5cdf) op, void *op_data)  
| Gathers data from a selection within a memory buffer raw data chunk in a dataset.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns the dataset access property list associated with a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dget_chunk_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaccff213d3e0765b86f66d08dd9959807) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fspace_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chk_idx, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, unsigned *filter_mask, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *addr, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)  
| Retrieves information about a chunk specified by its index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dget_chunk_info_by_coord](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, unsigned *filter_mask, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *addr, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)  
| Retrieves information about a chunk specified by its coordinates.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dget_chunk_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *chunk_bytes)  
| Returns the amount of storage allocated within the file for a raw data chunk in a dataset.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns an identifier for a copy of the dataset creation property list for a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dget_num_chunks](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8e15897dcc5799d6c09806644b492d7a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fspace_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *nchunks)  
| Retrieves number of chunks that have nonempty intersection with a specified selection.   
  
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) |  [H5Dget_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns dataset address in file.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns an identifier for a copy of the dataspace for a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dget_space_status](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [H5D_space_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a28e60d50e4eaeef27130829f66e39c7a) *allocation)  
| Determines whether space has been allocated for a dataset.   
  
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  [H5Dget_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns the amount of storage allocated for a dataset.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Returns an identifier for a copy of the datatype for a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Diterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga71421c684884ab49765748720fe938db) (void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [H5D_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a56f5174d268c404666360192432b13b9) op, void *operator_data)  
| Iterates over all selected elements in a dataspace.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dopen1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name)  
| Opens an existing dataset.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id)  
| Opens an existing dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf)  
| Reads raw data from a dataset into a provided buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread_chunk1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, uint32_t *filters, void *buf)  
| Reads a raw data chunk directly from a dataset in a file into a buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread_chunk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, uint32_t *filters, void *buf, size_t *buf_size)  
| Reads a raw data chunk directly from a dataset in a file into a buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915) (size_t count, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf[])  
| Reads raw data from a set of datasets into the provided buffers.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Drefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id)  
| Refreshes all buffers associated with a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dscatter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3525b15235ba1fd415f988899e48dc5c) ([H5D_scatter_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#ab2ee7d6aecf3e37b3f22fd5075ecfc00) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_space_id, void *dst_buf)  
| Scatters data into a selection within a memory buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dset_extent](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size[])  
| Changes the sizes of a dataset's dimensions.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dvlen_get_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)  
| Determines the number of bytes required to store variable-length (VL) data.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dvlen_reclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf)  
| Reclaims variable-length (VL) datatype memory buffers.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf)  
| Writes raw data from a buffer to a dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, uint32_t filters, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, size_t data_size, const void *buf)  
| Writes a raw data chunk from a buffer directly to a dataset in a file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dwrite_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8) (size_t count, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf[])  
| Writes raw data from a set buffers to a set of datasets.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)H5Dcreate
#define H5Dcreate [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)  
---  
[H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) is a macro that is mapped to either [H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.") or [H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file.").  


See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac482c2386aa3aea4c44730a627a7adb8)H5Dchunk_iter()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dchunk_iter  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | [H5D_chunk_iter_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a7a2008481c5cef463cbd0943a7042609) |  _cb_ ,   
|  | void * |  _op_data_ )  
Iterate over all chunks of a chunked dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | dxpl_id | Identifier of a transfer property list   
[in] | cb | User callback function, called for every chunk.   
[in] | op_data | User-defined pointer to data required by op 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
H5Dchunk_iter iterates over all chunks in the dataset, calling the user supplied callback with the details of the chunk and the supplied context `op_data`. 

Note
    Prior to HDF5 1.14.4, the address passed to the callback did not take the user block into account. 

Example
    For each chunk, print the allocated chunk size (0 for unallocated chunks). 
int
chunk_cb(const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, unsigned filter_mask, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, void *op_data)
{
// only print the allocated chunk size only
printf("%" [PRIuHSIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#af2a07cc002d935ff50b0045b439c5e84) "\n", size);
return EXIT_SUCCESS;
}
Iterate over all chunked datasets and chunks in a file. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5Ovisit_cb([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj, const char *name, const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *info, void *op_data)
{
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) retval = 0;
char *base_path = (char *)op_data;
if (info->[type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a6e12ef8203e400ee995a41a24e981539) == [H5O_TYPE_DATASET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdeca5ca744a77f8cd2b28dda90c37807ae31)) // current object is a dataset
{
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset, dcpl;
if ((dset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(obj, name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
retval = -1;
goto func_leave;
}
if ((dcpl = [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163)(dset)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
retval = -1;
goto fail_dcpl;
}
if ([H5Pget_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3)(dcpl) == [H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262)) // dataset is chunked
{
__label__ fail_dtype, fail_dspace, fail_shape;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dspace, dtype;
size_t size, i;
int rank;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) cdims[[H5S_MAX_RANK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a265cb2343f05cb71831119c90de31a8f)];
// get resources
if ((dtype = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)(dset)) < 0) {
retval = -1;
goto fail_dtype;
}
if ((dspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dset)) < 0) {
retval = -1;
goto fail_dspace;
}
// get the shape
if ((size = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(dtype)) == 0 || (rank = [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)(dspace)) < 0 ||
[H5Pget_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4ef814034f601f48ab1ed6db79b4354c)(dcpl, [H5S_MAX_RANK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a265cb2343f05cb71831119c90de31a8f), cdims) < 0) {
retval = -1;
goto fail_shape;
}
// calculate the nominal chunk size
size = 1;
for (i = 0; i < ([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd))rank; ++i)
size *= cdims[i];
// print dataset info
printf("%s%s : nominal chunk size %lu [B] \n", base_path, name, size);
// get the allocated chunk sizes
if ([H5Dchunk_iter](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac482c2386aa3aea4c44730a627a7adb8)(dset, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &chunk_cb, NULL) < 0) {
retval = -1;
goto fail_fig;
}
fail_fig:
fail_shape:
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(dspace);
fail_dspace:
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)(dtype);
fail_dtype:;
}
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dcpl);
fail_dcpl:
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
}
func_leave:
return retval;
} 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)H5Dclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Closes the specified dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.") terminates access to a dataset via the identifier `dset_id` and releases the underlying resources. 

Example
    
{
__label__ fail_dset, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset;
unsigned mode = [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394);
char file_name[] = "d1.h5";
// assume a priori knowledge of dataset name and size
char dset_name[] = "σύνολο/δεδομένων";
int elts[10];
if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_dset;
}
// read all dataset elements
if ([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), elts) < 0)
ret_val = EXIT_FAILURE;
// do something w/ the dataset elements
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
fail_dset:
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:;
} 

Since
    1.0.0 

See also
     [H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file."), [H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2)H5Dcreate1()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dcreate1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dcpl_id_ )  
Creates a dataset at the specified location. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Name of the dataset to create   
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier   
[in] | dcpl_id | Dataset creation property list identifier 

Returns
    Returns a dataset identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000022)**
    Superseded by [H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file.") or the macro [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf).  
[H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.") creates a data set with a name, `name`, in the location specified by the identifier `loc_id`. `loc_id` may be a file, group, dataset, named datatype or attribute. If an attribute, dataset, or named datatype is specified for `loc_id` then the dataset will be created at the location where the attribute, dataset, or named datatype is attached.
`name` can be a relative path based at `loc_id` or an absolute path from the root of the file. Use of this function requires that any intermediate groups specified in the path already exist.
The dataset's datatype and dataspace are specified by `type_id` and `space_id`, respectively. These are the datatype and dataspace of the dataset as it will exist in the file, which may differ from the datatype and dataspace in application memory.
Names within a group are unique: [H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.") will return an error if a link with the name specified in name already exists at the location specified in `loc_id`.
As is the case for any object in a group, the length of a dataset name is not limited.
`dcpl_id` is an [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102) property list created with `H5reate1()` and initialized with various property list functions described in Property List Interface.
[H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) and [H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") return an error if the dataset's datatype includes a variable-length (VL) datatype and the fill value is undefined, i.e., set to `NULL` in the dataset creation property list. Such a VL datatype may be directly included, indirectly included as part of a compound or array datatype, or indirectly included as part of a nested compound or array datatype.
[H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) and [H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") return a dataset identifier for success or a negative value for failure. The dataset identifier should eventually be closed by calling [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.") to release the resources it uses.
See [H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") for a discussion of the differences between [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) and [H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.").
The HDF5 library provides flexible means of specifying a fill value, of specifying when space will be allocated for a dataset, and of specifying when fill values will be written to a dataset. 

Version
    1.8.0 Function [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) renamed to [H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.") and deprecated in this release.  

Since
    1.0.0 

See also
     [H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset."), [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset."), [H5Tset_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c "Sets size for a datatype.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)H5Dcreate2()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dcreate2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ )  
Creates a new dataset and links it into the file. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Name of the dataset to create   
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier   
[in] | lcpl_id | Link creation property list identifier   
[in] | dcpl_id | Dataset creation property list identifier   
[in] | dapl_id | Dataset access property list identifier 

Returns
    Returns a dataset identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file.") creates a new dataset named `name` at the location specified by `loc_id`, and associates constant and initial persistent properties with that dataset, including the datatype `dtype_id`, the dataspace `space_id`, and other properties as specified by the dataset creation property list `dcpl_id` and the access property list `dapl_id`, respectively. Once created, the dataset is opened for access.
`loc_id` may specify a file, group, dataset, named datatype, or attribute. If an attribute, dataset, or named datatype is specified, then the dataset will be created at the location where the attribute, dataset, or named datatype is attached.
`name` may be either an absolute path in the file or a relative path from `loc_id` naming the dataset.
If `dtype_id` is a committed datatype, and if the file location associated with the committed datatype is different from the file location where the dataset will be created, the datatype is copied and converted to a transient type.
The link creation property list, `lcpl_id`, governs the creation of the link(s) by which the new dataset is accessed and the creation of any intermediate groups that may be missing.
The datatype and dataspace properties and the dataset creation and access property lists are attached to the dataset, so the caller may derive new datatypes, dataspaces, and creation and access properties from the old ones and reuse them in calls to create additional datasets. Once created, the dataset can be read from or written to. Reading data from a dataset that was not previously written, the HDF5 library will return default or user-defined fill values. 

Example
    
{
__label__ fail_lcpl, fail_dset, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, lcpl, fspace, dset;
unsigned mode = [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a);
char file_name[] = "d1.h5";
// link names can be arbitrary Unicode strings
char dset_name[] = "σύνολο/δεδομένων";
if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
if ((lcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_lcpl;
}
// use UTF-8 encoding for link names
if ([H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)(lcpl, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)) < 0) {
ret_val = EXIT_FAILURE;
goto fail_fspace;
}
// create intermediate groups as needed
if ([H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c)(lcpl, 1) < 0) {
ret_val = EXIT_FAILURE;
goto fail_fspace;
}
// create a 1D dataspace
if ((fspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){10}, NULL)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_fspace;
}
// create a 32-bit integer dataset
if ((dset = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file, dset_name, [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), fspace, lcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) ==
[H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_dset;
}
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
fail_dset:
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace);
fail_fspace:
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(lcpl);
fail_lcpl:
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:;
} 

Since
    1.8.0 

See also
     [H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset."), [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset."), [H5Tset_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c "Sets size for a datatype.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb)H5Dcreate_anon()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dcreate_anon  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ )  
Creates a dataset in a file without linking it into the file structure. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier   
[in] | dcpl_id | Dataset creation property list identifier   
[in] | dapl_id | Dataset access property list identifier 

Returns
    Returns a dataset identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") creates a dataset in the file specified by `loc_id`.
`loc_id` may specify a file, group, dataset, named datatype, or attribute. If an attribute, dataset, or named datatype is specified, then the dataset will be created at the location where the attribute, dataset, or named datatype is attached.
The dataset's datatype and dataspace are specified by `type_id` and `space_id`, respectively. These are the datatype and dataspace of the dataset as they will exist in the file, which may differ from the datatype and dataspace in application memory.
[H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") returns a new dataset identifier. Using this identifier, the new dataset must be linked into the HDF5 file structure with [H5Olink()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2c97dd58e64b67d16325fceb7e02113f "Creates a hard link to an object in an HDF5 file.") or it will be deleted when the file is closed. 

Since
    1.8.0 

See also
     [H5Olink()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga2c97dd58e64b67d16325fceb7e02113f "Creates a hard link to an object in an HDF5 file."), [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91)H5Dextend()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dextend  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _size_[] )  
Extends a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | size | Array containing the new size of each dimension 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000024)**
    Superseded by [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.").  
[H5Dextend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91 "Extends a dataset.") verifies that the dataset is at least of size `size`, extending it if necessary. The length of `size` is the same as that of the dataspace of the dataset being changed.
This function can be applied to the following datasets: 
  * Any dataset with unlimited dimensions 
  * A dataset with fixed dimensions if the current dimension sizes are less than the maximum sizes set with `maxdims` (see [H5Screate_simple()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae "Creates a new simple dataspace and opens it for access."))


Space on disk is immediately allocated for the new dataset extent if the dataset's space allocation time is set to [H5D_ALLOC_TIME_EARLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2a3c461ed83e0af151ef8c44ec232368b6). Fill values will be written to the dataset if the dataset's fill time is set to [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72) or [H5D_FILL_TIME_ALLOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aadd7bb84666434f7d1dc642e94c68eb28). (See [H5Pset_fill_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6bd822266b31f86551a9a1d79601b6a2 "Sets the time when fill values are written to a dataset.") and [H5Pset_alloc_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga85faefca58387bba409b65c470d7d851 "Sets the timing for storage space allocation.").)
This function ensures that the dataset dimensions are of at least the sizes specified in size. The function [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") must be used if the dataset dimension sizes are to be reduced. 

Version
    1.8.0 Function deprecated in this release. Parameter size syntax changed to `const hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size[]` in this release.  

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8d4a57e2b2b8c95cfecf6f75bdaa8343)H5Dfill()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dfill  | ( | const void * |  _fill_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fill_type_id_ ,   
|  | void * |  _buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _buf_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ )  
Fills dataspace elements with a fill value in a memory buffer. 
* * * 

Parameters
     [in] | fill | Pointer to the fill value to be used   
---|---|---  
[in] | fill_type_id | Fill value datatype identifier   
[in,out] | buf | Pointer to the memory buffer containing the selection to be filled   
[in] | buf_type_id | Datatype of dataspace elements to be filled   
[in] | space_id | Dataspace identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dfill()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8d4a57e2b2b8c95cfecf6f75bdaa8343 "Fills dataspace elements with a fill value in a memory buffer.") fills the dataspace selection, `space_id`, in memory with the fill value specified in `fill`. If `fill` is NULL, a fill value of 0 (zero) is used.
`fill_type_id` specifies the datatype of the fill value. `buf` specifies the buffer in which the fill elements will be written. `buf_type_id` specifies the datatype of those data elements. 

Note
    Note that if the fill value datatype differs from the memory buffer datatype, the fill value will be converted to the memory buffer datatype before filling the selection.      Applications sometimes write data only to portions of an allocated dataset. It is often useful in such cases to fill the unused space with a known fill value. 

See also
     [H5Pset_fill_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4 "Sets the fill value for a dataset."), [H5Pget_fill_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355 "Retrieves a dataset fill value."), [H5Pfill_value_defined()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga14f9bc2a0d6f9e62ab95661fc1045ad6 "Determines whether fill value is defined."), [H5Pset_fill_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6bd822266b31f86551a9a1d79601b6a2 "Sets the time when fill values are written to a dataset."), [H5Pget_fill_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga92c5eb5ee19bfd4a9184cf0428d1b00c "Retrieves the time when fill values are written to a dataset."), [H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class."), [H5Dcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga15a77e82383d821fee8ecbf9ab8408cb "Creates a dataset in a file without linking it into the file structure.") 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7)H5Dflush()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dflush  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Flushes all buffers associated with a dataset to disk. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7 "Flushes all buffers associated with a dataset to disk.") causes all buffers associated with a dataset to be immediately flushed to disk without removing the data from the cache. 

Note
    HDF5 does not possess full control over buffering. [H5Dflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7 "Flushes all buffers associated with a dataset to disk.") flushes the internal HDF5 buffers and then asks the operating system (the OS) to flush the system buffers for the open files. After that, the OS is responsible for ensuring that the data is actually flushed to disk. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be)H5Dgather()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dgather  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_space_id_ ,   
---|---|---|---  
|  | const void * |  _src_buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | size_t |  _dst_buf_size_ ,   
|  | void * |  _dst_buf_ ,   
|  | [H5D_gather_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#ab0c47a79dc28343e9c1288e911ae5cdf) |  _op_ ,   
|  | void * |  _op_data_ )  
Gathers data from a selection within a memory buffer raw data chunk in a dataset. 
* * * 

Parameters
     [in] | src_space_id | Dataspace identifier for the source buffer   
---|---|---  
[in] | src_buf | Source buffer which the data will be gathered from   
[in] | type_id | Datatype identifier for the source   
[in] | dst_buf_size | Size in bytes of `dst_buf`  
[out] | dst_buf | Destination buffer for the gathered data   
[in] | op | Callback function which handles the gathered data   
[in] | op_data | User-defined pointer to data required by `op` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dgather()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be "Gathers data from a selection within a memory buffer raw data chunk in a dataset.") retrieves data from a selection within the supplied buffer src_buf and passes it to the supplied callback function `op` in a contiguous form.
The dataspace `src_space_id` describes both the dimensions of the source buffer and the selection within the source buffer to gather data from.
`src_buf` must be at least the size of the gathered data, that is, the number of elements in the extent of `src_space_id` times the size in bytes of `type_id`.
The datatype `type_id` describes the data in both the source and destination buffers. This information is used to calculate the element size.
The data is gathered into `dst_buf`, which needs to be large enough to hold all the data if the callback function `op` is not provided.
`op` is a callback function that handles the gathered data. It is optional if `dst_buf` is large enough to hold all of the gathered data; required otherwise.
If no callback function is provided, [H5Dgather()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be "Gathers data from a selection within a memory buffer raw data chunk in a dataset.") simply gathers the data into `dst_buf` and returns. If a callback function is provided, [H5Dgather()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga1f6a428a8234d7c2ccba7da4742d79be "Gathers data from a selection within a memory buffer raw data chunk in a dataset.") repeatedly gathers up to `dst_buf_size` bytes to process the serialized data.
The callback function `op` should process, store, or otherwise, make use of the data returned in `dst_buf` before it returns, because the buffer will be overwritten unless it is the last call to the callback. This function will be repeatedly called until all gathered elements have been passed to the callback in `dst_buf`. The callback function should return zero (0) to indicate success, and a negative value to indicate failure. 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b)H5Dget_access_plist()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dget_access_plist  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns the dataset access property list associated with a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a dataset access property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dget_access_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b "Returns the dataset access property list associated with a dataset.") returns a copy of the dataset access property list used to open the specified dataset, `dset_id`. Modifications to the returned property list will have no effect on the dataset it was retrieved from.
The chunk cache parameters in the returned property lists will be those used by the dataset. If the properties in the file access property list were used to determine the dataset's chunk cache configuration, then those properties will be present in the returned dataset access property list. If the dataset does not use a chunked layout, then the chunk cache properties will be set to the default. The chunk cache properties in the returned list are considered to be “set”, and any use of this list will override the corresponding properties in the file's file access property list.
All link access properties in the returned list will be set to the default values. 

Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaccff213d3e0765b86f66d08dd9959807)H5Dget_chunk_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dget_chunk_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fspace_id_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _chk_idx_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | unsigned * |  _filter_mask_ ,   
|  |  [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _addr_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _size_ )  
Retrieves information about a chunk specified by its index. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | fspace_id | File dataspace selection identifier (See Note below)   
[in] | chk_idx | Index of the chunk   
[out] | offset | Logical position of the chunk's first element in units of dataset elements   
[out] | filter_mask | Bitmask indicating the filters used when the chunk was written   
[out] | addr | Chunk address in the file, taking the user block (if any) into account   
[out] | size | Chunk size in bytes, 0 if the chunk does not exist 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dget_chunk_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaccff213d3e0765b86f66d08dd9959807 "Retrieves information about a chunk specified by its index.") retrieves the offset coordinates, `offset`, filter mask, `filter_mask`, size, `size`, and address `addr` for the dataset specified by the identifier `dset_id` and the chunk specified by the index `index`. The chunk belongs to a set of chunks in the selection specified by `fspace_id`. If the queried chunk does not exist in the file, the size will be set to 0 and address to [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c). The value pointed to by filter_mask will not be modified. `NULL` can be passed in for any `out` parameters. 

Note
    Prior to HDF5 1.14.4, the reported address did not take the user block into account.
`chk_idx` is the chunk index in the selection. The index value may have a value of 0 up to the number of chunks stored in the file that has a nonempty intersection with the file dataspace selection. 

Note
    As of 1.10.5, the dataspace intersection is not yet supported. Hence, the index is of all the written chunks.
`fspace_id` specifies the file dataspace selection. It is intended to take [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) to specify the current selection. 

Note
    Please be aware that this function currently does not support non-trivial selections; thus `fspace_id` has no effect. Also, the implementation does not handle the [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) macro correctly. As a workaround, an application can get the dataspace for the dataset using [H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.") and pass that in for `fspace_id`. This will be fixed in a future release. 

Since
    1.10.5 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0)H5Dget_chunk_info_by_coord()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dget_chunk_info_by_coord  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | unsigned * |  _filter_mask_ ,   
|  |  [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _addr_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _size_ )  
Retrieves information about a chunk specified by its coordinates. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | offset | Logical position of the chunk's first element in units of dataset elements   
[out] | filter_mask | Bitmask indicating the filters used when the chunk was written   
[out] | addr | Chunk address in the file, taking the user block (if any) into account   
[out] | size | Chunk size in bytes, 0 if the chunk does not exist 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dget_chunk_info_by_coord()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0 "Retrieves information about a chunk specified by its coordinates.") retrieves the `filter_mask`, `size`, and `addr` for a chunk in the dataset specified by `dset_id`, using the coordinates specified by `offset`.
If the queried chunk does not exist in the file, `size` will be set to 0, `addr` to `HADDR_UNDEF`, and the buffer `filter_mask` will not be modified.
`offset` is a pointer to a one-dimensional array with a size equal to the dataset's rank. Each element is the logical position of the chunk's first element in a dimension. 

Note
    Prior to HDF5 1.14.4, the reported address did not take the user block into account. 

Since
    1.10.5 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234)H5Dget_chunk_storage_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dget_chunk_storage_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _chunk_bytes_ )  
Returns the amount of storage allocated within the file for a raw data chunk in a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | offset | Logical offset in the dataset for the chunk to query   
[out] | chunk_bytes | The size in bytes for the chunk 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dget_chunk_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234 "Returns the amount of storage allocated within the file for a raw data chunk in a dataset.") returns the size in bytes allocated in the file for a raw data chunk as specified by its logical `offset` in the dataset `dset_id`. The size is returned in `chunk_nbytes`. It is the size of the compressed data if the chunk is filtered and the size may be zero if no storage is allocated yet for the dataset. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163)H5Dget_create_plist()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dget_create_plist  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns an identifier for a copy of the dataset creation property list for a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a dataset creation property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163 "Returns an identifier for a copy of the dataset creation property list for a dataset.") returns an identifier for a copy of the dataset creation property list associated with the dataset specified by `dset_id`.
The creation property list identifier should be released with [H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") to prevent resource leaks. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8e15897dcc5799d6c09806644b492d7a)H5Dget_num_chunks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dget_num_chunks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fspace_id_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _nchunks_ )  
Retrieves number of chunks that have nonempty intersection with a specified selection. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | fspace_id | File dataspace selection identifier   
[out] | nchunks | Number of chunks in the selection 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dget_num_chunks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8e15897dcc5799d6c09806644b492d7a "Retrieves number of chunks that have nonempty intersection with a specified selection.") retrieves the number of chunks nchunks in a set of selected elements specified by `fspace_id` for a dataset specified by the identifier `dset_id`. If `fspace_id` is [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), the function will retrieve the total number of chunks stored for the dataset.
`fspace_id` specifies the file dataspace selection. It is intended to take [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) for specifying the current selection. 

Note
    Please be aware that this function currently does not support non-trivial selections, thus `fspace_id` has no effect. Also, the implementation does not handle the [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) macro correctly. As a workaround, application can get the dataspace for the dataset using [H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.") and pass that in for `fspace_id`. This will be fixed in a future release. 

Since
    1.10.5 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3)H5Dget_offset()
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) H5Dget_offset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns dataset address in file. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns the offset in bytes; otherwise, returns [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c), a negative value.  
[H5Dget_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga70ce7ab523b06c6c6a93fb28e916c2b3 "Returns dataset address in file.") returns the address in the file of the dataset, `dset_id`. That address is expressed as the offset in bytes from the beginning of the file. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)H5Dget_space()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dget_space  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns an identifier for a copy of the dataspace for a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a dataspace identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.") makes a copy of the dataspace of the dataset specified by `dset_id`. The function returns an identifier for the new copy of the dataspace.
A dataspace identifier returned from this function should be released with [H5Sclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1 "Releases and terminates access to a dataspace.") when the identifier is no longer needed so that resource leaks will not occur. 

Since
    1.0.0 

Example
    
{
__label__ fail_update, fail_fspace, fail_dset, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset, fspace;
unsigned mode = [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4);
char file_name[] = "d1.h5";
char dset_name[] = "σύνολο/δεδομένων";
int new_elts[6][2] = {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}};
if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_dset;
}
// get the dataset's dataspace
if ((fspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dset)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_fspace;
}
// select the first 5 elements in odd positions
if ([H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){1}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){2}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){5},
NULL) < 0) {
ret_val = EXIT_FAILURE;
goto fail_update;
}
// (implicitly) select and write the first 5 elements of the second column of NEW_ELTS
if ([H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), fspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), new_elts) < 0)
ret_val = EXIT_FAILURE;
fail_update:
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace);
fail_fspace:
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
fail_dset:
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:;
} 

See also
     [H5Sclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1 "Releases and terminates access to a dataspace.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c)H5Dget_space_status()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dget_space_status  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  |  [H5D_space_status_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a28e60d50e4eaeef27130829f66e39c7a) * |  _allocation_ )  
Determines whether space has been allocated for a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[out] | allocation | Space allocation status 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dget_space_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.") determines whether space has been allocated for the dataset `dset_id`. 

Note
     **BUG:** Prior to the HDF5 1.14.0, 1.12.2 and 1.10.9 releases, [H5Dget_space_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.") may return incorrect space allocation status values for datasets with filters applied to them. [H5Dget_space_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.") calculated the space allocation status by comparing the sum of the sizes of all the allocated chunks in the dataset against the total data size of the dataset, as calculated by the number of elements in the dataset's dataspace multiplied by the dataset's datatype size. If the dataset had any compression filters applied to it and the dataset chunks were successfully compressed, the sum of the sizes of the allocated dataset chunks would generally always be less than the total data size of the dataset, and [H5Dget_space_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7639ef5c12cb906c71670ce73b856a4c "Determines whether space has been allocated for a dataset.") wouldn't ever return `H5D_SPACE_STATUS_ALLOCATED`. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00)H5Dget_storage_size()
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) H5Dget_storage_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns the amount of storage allocated for a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns the amount of storage space, in bytes, or 0 (zero).  
[H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.") returns the amount of storage, in bytes, that is allocated in the file for the raw data of the dataset specified by `dset_id`. [H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.") reports only the space required to store the dataset elements, excluding any metadata. 
  * For contiguous datasets, the returned size equals the current allocated size of the raw data. 
  * For unfiltered chunked datasets, the returned size is the number of allocated chunks times the chunk size. 
  * For filtered chunked datasets, the returned size is the space required to store the filtered data. For example, if a compression filter is in use, [H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.") will return the total space required to store the compressed chunks.



Note
    Note that [H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.") is not generally an appropriate function to use when determining the amount of memory required to work with a dataset. In such circumstances, you must determine the number of data points in a dataset and the size of an individual dataset element. [H5Sget_simple_extent_npoints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga9d369fe1e01a95710d320bc5fd32cf83 "Determines the number of elements in a dataspace.") and [H5Tget_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e "Returns the size of a datatype.") can be used to calculate that amount. 

Warning
     [H5Dget_storage_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gafb249479a493e80891f0c7f5d8a91b00 "Returns the amount of storage allocated for a dataset.") does not differentiate between 0 (zero), the value returned for the storage size of a dataset with no stored values, and 0 (zero), the value returned to indicate an error. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)H5Dget_type()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dget_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Returns an identifier for a copy of the datatype for a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.") returns an identifier of a copy of the datatype for a dataset.
If a dataset has a named datatype, then an identifier to the opened datatype is returned. Otherwise, the returned datatype is read-only. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga71421c684884ab49765748720fe938db)H5Diterate()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Diterate  | ( | void * |  _buf_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [H5D_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a56f5174d268c404666360192432b13b9) |  _op_ ,   
|  | void * |  _operator_data_ )  
Iterates over all selected elements in a dataspace. 
* * * 

Parameters
     [in,out] | buf | Buffer containing the elements to iterate over   
---|---|---  
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier   
[in] | op | Function pointer   
[in,out] | operator_data | User-defined data 

Returns
     **Success:** The return value of the first operator that returns non-zero, or zero if all members were processed with no operator returning non-zero.       **Failure:** Negative if an error occurs in the library, or the negative value returned by one of the operators.  
[H5Diterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga71421c684884ab49765748720fe938db "Iterates over all selected elements in a dataspace.") iterates over all the data elements in the memory buffer `buf`, executing the callback function `op` once for each such data element. 

Attention
    Unlike other HDF5 iterators, this iteration operation cannot be restarted at the point of exit; a second [H5Diterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga71421c684884ab49765748720fe938db "Iterates over all selected elements in a dataspace.") call will always restart at the beginning. 

Warning
    Modifying the selection of `space_id` during iteration will lead to undefined behavior. 

Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81)H5Dopen1()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dopen1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ )  
Opens an existing dataset. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Name of the dataset to access 

Returns
    Returns a dataset identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000023)**
    Superseded by [H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset.") or the macro [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3).  
[H5Dopen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81 "Opens an existing dataset.") opens an existing dataset for access at the location specified by `loc_id`. `loc_id` may be a file, group, dataset, named datatype or attribute. If an attribute, dataset, or named datatype is specified for loc_id then the dataset will be opened at the location where the attribute, dataset, or named datatype is attached. name is a dataset name and is used to identify the dataset in the file.
A dataset opened with this function should be closed with [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.") when the dataset is no longer needed so that resource leaks will not develop. 

Version
    1.8.0 Function [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) renamed to [H5Dopen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81 "Opens an existing dataset.") and deprecated in this release.  

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)H5Dopen2()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dopen2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ )  
Opens an existing dataset. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Name of the dataset to open   
[in] | dapl_id | Dataset access property list identifier 

Returns
    Returns a dataset identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset.") opens the existing dataset specified by a location identifier and name, `loc_id` and `name`, respectively.
`loc_id` may specify a file, group, dataset, named datatype, or attribute. If an attribute, dataset, or named datatype is specified then the dataset will be opened at the location where the attribute, dataset, or named datatype is attached. 

Since
    1.8.0 

See also
     [H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file."), [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)H5Dread()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | void * |  _buf_ )  
Reads raw data from a dataset into a provided buffer. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier Identifier of the dataset to read from   
---|---|---  
[in] | mem_type_id | Identifier of the memory datatype   
[in] | mem_space_id | Identifier of the memory dataspace   
[in] | file_space_id | Identifier of the dataset's dataspace in the file   
[in] | dxpl_id | Identifier of a transfer property list   
[out] | buf | Buffer to receive data read from file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") reads a dataset, specified by its identifier `dset_id`, from the file into an application memory buffer `buf`. Data transfer properties are defined by the argument `dxpl_id`. The memory datatype of the (partial) dataset is identified by the identifier `mem_type_id`. The part of the dataset to read is defined by `mem_space_id` and `file_space_id`.
`file_space_id` is used to specify only the selection within the file dataset's dataspace. Any dataspace specified in `file_space_id` is ignored by the library and the dataset's dataspace is always used. `file_space_id` can be the constant [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), which indicates that the entire file dataspace, as defined by the current dimensions of the dataset, is to be selected.
`mem_space_id` is used to specify both the memory dataspace and the selection within that dataspace. `mem_space_id` can be the constant [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), in which case the file dataspace is used for the memory dataspace and the selection defined with `file_space_id` is used for the selection within that dataspace.
The number of elements selected in the memory dataspace _must_ be equal to the number of elements selected in the file dataspace.
The behavior of the library for the various combinations of valid dataspace identifiers and [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) for the `mem_space_id` and the `file_space_id` parameters is described below:
mem_space_id  | file_space_id  | Behavior   
---|---|---  
valid dataspace ID  | valid dataspace ID  |  `mem_space_id` specifies the memory dataspace and the selection within it. `file_space_id` specifies the selection within the file dataset's dataspace.   
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) | valid dataspace ID  | The file dataset's dataspace is used for the memory dataspace and the selection specified with `file_space_id` specifies the selection within it. The combination of the file dataset's dataspace and the selection from `file_space_id` is used for memory also.   
valid dataspace ID  |  [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) |  `mem_space_id` specifies the memory dataspace and the selection within it. The selection within the file dataset's dataspace is set to the "all" selection.   
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) |  [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) | The file dataset's dataspace is used for the memory dataspace and the selection within the memory dataspace is set to the "all" selection. The selection within the file dataset's dataspace is set to the "all" selection.  

Note
    If no storage space was allocated for the dataset and a fill value is defined, the returned buffer `buf` is filled with the fill value. 

Example
      
{
__label__ fail_dset, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset;
unsigned mode = [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394);
char file_name[] = "d1.h5";
// assume a priori knowledge of dataset name and size
char dset_name[] = "σύνολο/δεδομένων";
int elts[10];
if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_dset;
}
// read all dataset elements
if ([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), elts) < 0)
ret_val = EXIT_FAILURE;
// do something w/ the dataset elements
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
fail_dset:
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:;
} 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3)H5Dread_chunk1()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread_chunk1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | uint32_t * |  _filters_ ,   
|  | void * |  _buf_ )  
Reads a raw data chunk directly from a dataset in a file into a buffer. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | offset | Logical position of the chunk's first element in the dataspace   
[in,out] | filters | Mask for identifying the filters in use   
[out] | buf | Buffer to receive data read from the chunk 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000026)**
    Superseded by [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") or the macro [H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e).  
[H5Dread_chunk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.") reads a raw data chunk as specified by its logical offset `offset` in a chunked dataset `dset_id` from the dataset in the file into the application memory buffer `buf`. The data in `buf` is read directly from the file bypassing the library's internal data transfer pipeline, including filters.
`offset` is an array specifying the logical position of the first element of the chunk in the dataset's dataspace. The length of the `offset` array must equal the number of dimensions, or rank, of the dataspace. The values in `offset` must not exceed the dimension limits and must specify a point that falls on a dataset chunk boundary.
The mask `filters` indicates which filters were used when the chunk was written. A zero value (all bits 0) indicates that all enabled filters are applied on the chunk. A filter is skipped if the bit corresponding to the filter's position in the pipeline (0 ≤ position < 32) is turned on.
`buf` is the memory buffer containing the chunk read from the dataset in the file. 

Attention
    It is strongly recommended to use [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") instead of this function due to the potential for memory corruption. During the typical usage pattern of this function, the application has no way of knowing the size of the chunk on disk, or even a maximum size (filters can increase the size of a chunk). The library also has no way of knowing the size of `buf`, so there is a potential for the library to write past the end of the buffer. The only general way to avoid this problem, besides upgrading to [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer."), is to query the chunk size beforehand, which can be expensive.      Exercise caution when using [H5Dread_chunk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file."), as they read and write data chunks directly in a file. [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") bypasses hyperslab selection, the conversion of data from one datatype to another, and the filter pipeline to write the chunk. Developers should have experience with these processes before using this function. Please see [Using the Direct Chunk Write Function](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#subsec_hldo_direct_chunk_using) for more information. 

Note
     [H5Dread_chunk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") are currently not supported with parallel HDF5 and do not support variable-length datatypes. 

Version
    2.0.0 Function was deprecated 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c)H5Dread_chunk2()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread_chunk2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | uint32_t * |  _filters_ ,   
|  | void * |  _buf_ ,   
|  | size_t * |  _buf_size_ )  
Reads a raw data chunk directly from a dataset in a file into a buffer. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | offset | Logical position of the chunk's first element in the dataspace   
[in,out] | filters | Mask for identifying the filters in use   
[out] | buf | Buffer to receive data read from the chunk   
[in,out] | buf_size | Size of buf in bytes 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") reads a raw data chunk as specified by its logical offset `offset` in a chunked dataset `dset_id` from the dataset in the file into the application memory buffer `buf`. The data in `buf` is read directly from the file bypassing the library's internal data transfer pipeline, including filters.
`offset` is an array specifying the logical position of the first element of the chunk in the dataset's dataspace. The length of the `offset` array must equal the number of dimensions, or rank, of the dataspace. The values in `offset` must not exceed the dimension limits and must specify a point that falls on a dataset chunk boundary.
The mask `filters` indicates which filters were used when the chunk was written. A zero value (all bits 0) indicates that all enabled filters are applied on the chunk. A filter is skipped if the bit corresponding to the filter's position in the pipeline (0 ≤ position < 32) is turned on.
`buf` is the memory buffer containing the chunk read from the dataset in the file.
`buf_size` must be passed as a pointer to a variable holding the allocated size, in bytes, of the memory buffer `buf`. On exit, `*buf_size` is set to the buffer size needed to read the chunk, which is the same as the size of the chunk on disk. If the value of `*buf_size` passed in was insufficient to read the entire, chunk, no data is read. `buf` may be passed as NULL as long as `*buf_size` is 0. `filters` is always set by this function even if the chunk was not read. 

Attention
    Exercise caution when using [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file."), as they read and write data chunks directly in a file. [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") bypasses hyperslab selection, the conversion of data from one datatype to another, and the filter pipeline to write the chunk. Developers should have experience with these processes before using this function. Please see [Using the Direct Chunk Write Function](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#subsec_hldo_direct_chunk_using) for more information. 

Note
     [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") are currently not supported with parallel HDF5 and do not support variable-length datatypes. 

Since
    2.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915)H5Dread_multi()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread_multi  | ( | size_t |  _count_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | void * |  _buf_[] )  
Reads raw data from a set of datasets into the provided buffers. 
* * * 

Parameters
     [in] | count | Number of datasets to read from   
---|---|---  
[in] | dset_id | Identifiers of the datasets to read from   
[in] | mem_type_id | Identifiers of the memory datatypes   
[in] | mem_space_id | Identifiers of the memory dataspaces   
[in] | file_space_id | Identifiers of the datasets' dataspaces in the file   
[in] | dxpl_id | Identifier of a transfer property list   
[out] | buf | Buffers to receive data read from file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dread_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915 "Reads raw data from a set of datasets into the provided buffers.") reads data from `count` datasets, whose identifiers are listed in the `dset_id` array, from the file into multiple application memory buffers listed in the `buf` array. Data transfer properties are defined by the argument `dxpl_id`. The memory datatypes of each dataset are listed by identifier in the `mem_type_id` array. The parts of each dataset to read are listed by identifier in the `file_space_id` array, and the parts of each application memory buffer to read to are listed by identifier in the `mem_space_id` array. All array parameters have length `count`.
This function will produce the same results as `count` calls to [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer."). Information listed in that function about the specifics of its behavior also applies to [H5Dread_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915 "Reads raw data from a set of datasets into the provided buffers."). By calling [H5Dread_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915 "Reads raw data from a set of datasets into the provided buffers.") instead of multiple calls to [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer."), however, the library can in some cases pass information about the entire I/O operation to the file driver, which can improve performance.
All datasets must be in the same HDF5 file, and each unique dataset may only be listed once. If this function is called collectively in parallel, each rank must pass exactly the same list of datasets in `dset_id` , though the other parameters may differ. 

Since
    1.14.0 

See also
     [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da)H5Drefresh()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Drefresh  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _dset_id_ | ) |   
---|---|---|---|---|---  
Refreshes all buffers associated with a dataset. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Drefresh()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3c1ea7e5db3f62d9cf03dd62d1fb08da "Refreshes all buffers associated with a dataset.") causes all buffers associated with a dataset to be cleared and immediately re-loaded with updated contents from disk.
This function essentially closes the dataset, evicts all metadata associated with it from the cache, and then re-opens the dataset. The reopened dataset is automatically re-registered with the same identifier. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3525b15235ba1fd415f988899e48dc5c)H5Dscatter()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dscatter  | ( | [H5D_scatter_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#ab2ee7d6aecf3e37b3f22fd5075ecfc00) |  _op_ ,   
---|---|---|---  
|  | void * |  _op_data_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_space_id_ ,   
|  | void * |  _dst_buf_ )  
Scatters data into a selection within a memory buffer. 
* * * 

Parameters
     [in] | op | Callback function which provides data to be scattered   
---|---|---  
[in] | op_data | User-defined pointer to data required by op   
[in] | type_id | Identifier for the datatype describing the data in both the source and destination buffers   
[in] | dst_space_id | Identifier for the dataspace for destination   
[out] | dst_buf | Destination buffer which the data will be scattered to 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dscatter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3525b15235ba1fd415f988899e48dc5c "Scatters data into a selection within a memory buffer.") retrieves data from the supplied callback `op` and scatters it to the supplied buffer `dst_buf` in a manner similar to data being written to a dataset.
`dst_space_id` is a dataspace that defines the extent of `dst_buf` and the selection within it to scatter the data to.
`type_id` is the datatype of the data to be scattered in both the source and destination buffers.
`dst_buf` must be at least as large as the number of elements in the extent of `dst_space_id` times the size in bytes of `type_id`.
To retrieve the data to be scattered, [H5Dscatter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga3525b15235ba1fd415f988899e48dc5c "Scatters data into a selection within a memory buffer.") repeatedly calls `op`, which should return a valid source buffer, until enough data to fill the selection has been retrieved. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f)H5Dset_extent()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dset_extent  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _size_[] )  
Changes the sizes of a dataset's dimensions. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | size[] | Array containing the new magnitude of each dimension of the dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") sets the current dimensions of the chunked dataset `dset_id` to the sizes specified in size.
`size` is a 1-dimensional array with n elements, where `n` is the rank of the dataset's current dataspace.
This function can be applied to the following datasets:
  * A chunked dataset with unlimited dimensions
  * A chunked dataset with fixed dimensions if the new dimension sizes are less than the maximum sizes set with maxdims (see [H5Screate_simple()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae "Creates a new simple dataspace and opens it for access."))
  * An external dataset with unlimited dimensions
  * An external dataset with fixed dimensions if the new dimension sizes are less than the maximum sizes set with `maxdims`


Note that external datasets are always contiguous and can be extended only along the first dimension.
Space on disk is immediately allocated for the new dataset extent if the dataset's space allocation time is set to [H5D_ALLOC_TIME_EARLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2a3c461ed83e0af151ef8c44ec232368b6).
Fill values will be written to the dataset in either of the following situations, but not otherwise:
  * If the dataset's fill time is set to [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72) and a fill value is defined (see [H5Pset_fill_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6bd822266b31f86551a9a1d79601b6a2 "Sets the time when fill values are written to a dataset.") and [H5Pset_fill_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4 "Sets the fill value for a dataset."))
  * If the dataset's fill time is set to [H5D_FILL_TIME_ALLOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aadd7bb84666434f7d1dc642e94c68eb28) (see [H5Pset_alloc_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga85faefca58387bba409b65c470d7d851 "Sets the timing for storage space allocation."))



Note
    If the sizes specified in `size` array are smaller than the dataset's current dimension sizes, [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") will reduce the dataset's dimension sizes to the specified values. It is the user application's responsibility to ensure that valuable data is not lost as [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") does not check.      Except for external datasets, [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") is for use with chunked datasets only, not contiguous datasets.      A call to [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") affects the dataspace of a dataset. If a dataspace handle was opened for a dataset prior to a call to [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.") then that dataspace handle will no longer reflect the correct dataspace extent of the dataset. [H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.") must be called (after closing the previous handle) to obtain the current dataspace extent. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76)H5Dvlen_get_buf_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dvlen_get_buf_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _size_ )  
Determines the number of bytes required to store variable-length (VL) data. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | type_id | Datatype identifier   
[in] | space_id | Dataspace identifier   
[out] | size | Size in bytes of the memory buffer required to store the VL data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dvlen_get_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76 "Determines the number of bytes required to store variable-length \(VL\) data.") determines the number of bytes required to store the VL data from the dataset, using `space_id` for the selection in the dataset on disk and the `type_id` for the memory representation of the VL data in memory. `size` is returned with the number of bytes required to store the VL data in memory. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119)H5Dvlen_reclaim()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dvlen_reclaim  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | void * |  _buf_ )  
Reclaims variable-length (VL) datatype memory buffers. 
* * * 

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | space_id | Dataspace identifier   
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | buf | Pointer to the buffer to be reclaimed 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000025)**
    Superseded by [H5Treclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.").  
[H5Dvlen_reclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119 "Reclaims variable-length \(VL\) datatype memory buffers.") reclaims memory buffers created to store VL datatypes.
The `type_id` must be the datatype stored in the buffer. The `space_id` describes the selection for the memory buffer to free the VL datatypes within. The `dxpl_id` is the dataset transfer property list that was used for the I/O transfer to create the buffer. And `buf` is the pointer to the buffer to be reclaimed.
The VL structures ([hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html)) in the user's buffer are modified to zero out the VL information after the memory has been reclaimed.
If nested VL datatypes were used to create the buffer, this routine frees them from the bottom up, releasing all the memory without creating memory leaks. 

Version
    1.12.0 Function was deprecated 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)H5Dwrite()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dwrite  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const void * |  _buf_ )  
Writes raw data from a buffer to a dataset. 
* * * 

Parameters
     [in] | dset_id | Identifier of the dataset to read from   
---|---|---  
[in] | mem_type_id | Identifier of the memory datatype   
[in] | mem_space_id | Identifier of the memory dataspace   
[in] | file_space_id | Identifier of the dataset's dataspace in the file   
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | buf | Buffer with data to be written to the file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") writes a (partial) dataset, specified by its identifier `dset_id`, from the application memory buffer `buf` into the file. Data transfer properties are defined by the argument `dxpl_id`. The memory datatype of the (partial) dataset is identified by the identifier `mem_type_id`. The part of the dataset to write is defined by `mem_space_id` and `file_space_id`.
If `mem_type_id` is either a fixed-length or variable-length string, it is important to set the string length when defining the datatype. String datatypes are derived from [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) (or [H5T_FORTRAN_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga62335f6f57c2809fa1b3b1f9472eb2f6) for Fortran codes), which defaults to 1 character in size. See [H5Tset_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c "Sets size for a datatype.") and Creating variable-length string datatypes.
`file_space_id` is used to specify only the selection within the file dataset's dataspace. Any dataspace specified in `file_space_id` is ignored by the library and the dataset's dataspace is always used. `file_space_id` can be the constant [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), which indicates that the entire file dataspace, as defined by the current dimensions of the dataset, is to be selected.
`mem_space_id` is used to specify both the memory dataspace and the selection within that dataspace. mem_space_id can be the constant [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), in which case the file dataspace is used for the memory dataspace and the selection defined with `file_space_id` is used for the selection within that dataspace.
The behavior of the library for the various combinations of valid dataspace IDs and [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) for the mem_space_id and thefile_space_id parameters is described below:
`mem_space_id` |  `file_space_id` | Behavior   
---|---|---  
valid dataspace ID  | valid dataspace ID  |  `mem_space_id` specifies the memory dataspace and the selection within it. `file_space_id` specifies the selection within the file dataset's dataspace.   
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) | valid dataspace ID  | The file dataset's dataspace is used for the memory dataspace and the selection specified with `file_space_id` specifies the selection within it. The combination of the file dataset's dataspace and the selection from `file_space_id` is used for memory also. valid dataspace ID   
valid dataspace ID  |  [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) |  `mem_space_id` specifies the memory dataspace and the selection within it. The selection within the file dataset's dataspace is set to "all" selection.   
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) |  [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) | The file dataset's dataspace is used for the memory dataspace and the selection within the memory dataspace is set to the "all" selection. The selection within the file dataset's dataspace is set to the "all" selection.   
Setting an "all" selection indicates that the entire dataspace, as defined by the current dimensions of a dataspace, will be selected. The number of elements selected in the memory dataspace must match the number of elements selected in the file dataspace.
`dxpl_id` can be the constant [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), in which case the default data transfer properties are used.
Writing to a dataset will fail if the HDF5 file was not opened with write access permissions.
If the dataset's space allocation time is set to [H5D_ALLOC_TIME_LATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2a7c5fcb6f9c8adecf455939c3d625b7e8) or [H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636) and the space for the dataset has not yet been allocated, that space is allocated when the first raw data is written to the dataset. Unused space in the dataset will be written with fill values at the same time if the dataset's fill time is set to [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72) or [H5D_FILL_TIME_ALLOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aadd7bb84666434f7d1dc642e94c68eb28). 

Attention
    If you are planning to use compression with parallel HDF5, ensure that calls to [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") occur in collective mode. In other words, all MPI ranks (in the relevant communicator) call [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") and pass a dataset transfer property list with the MPI-IO collective option property set to [H5FD_MPIO_COLLECTIVE_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#afaf7d5667632176e8daca47549e29fb8a3816f1011586f6f6f57ce6c2a6c2fcbe).  
Note that data transformations are currently **not** supported when writing to datasets in parallel and with compression enabled.      If a dataset's storage layout is 'compact', care must be taken when writing data to the dataset in parallel. A compact dataset's raw data is cached in memory and may be flushed to the file from any of the parallel processes, so parallel applications should always attempt to write identical data to the dataset from all processes. 

Example
    
{
__label__ fail_update, fail_fspace, fail_dset, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dset, fspace;
unsigned mode = [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4);
char file_name[] = "d1.h5";
char dset_name[] = "σύνολο/δεδομένων";
int new_elts[6][2] = {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}};
if ((file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
if ((dset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, dset_name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_dset;
}
// get the dataset's dataspace
if ((fspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dset)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_fspace;
}
// select the first 5 elements in odd positions
if ([H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){1}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){2}, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){5},
NULL) < 0) {
ret_val = EXIT_FAILURE;
goto fail_update;
}
// (implicitly) select and write the first 5 elements of the second column of NEW_ELTS
if ([H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), fspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), new_elts) < 0)
ret_val = EXIT_FAILURE;
fail_update:
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace);
fail_fspace:
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset);
fail_dset:
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:;
} 

Since
    1.0.0 

See also
     [H5Pset_fill_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6bd822266b31f86551a9a1d79601b6a2 "Sets the time when fill values are written to a dataset."), [H5Pset_alloc_time()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga85faefca58387bba409b65c470d7d851 "Sets the timing for storage space allocation.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109)H5Dwrite_chunk()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dwrite_chunk  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | uint32_t |  _filters_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | size_t |  _data_size_ ,   
|  | const void * |  _buf_ )  
Writes a raw data chunk from a buffer directly to a dataset in a file. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | filters | Mask for identifying the filters in use   
[in] | offset | Logical position of the chunk's first element in the dataspace   
[in] | data_size | Size of the actual data to be written in bytes   
[in] | buf | Buffer containing data to be written to the chunk 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") writes a raw data chunk as specified by its logical offset `offset` in a chunked dataset `dset_id` from the application memory buffer `buf` to the dataset in the file. Typically, the data in `buf` is preprocessed in memory by a custom transformation, such as compression. The chunk will bypass the library's internal data transfer pipeline, including filters, and will be written directly to the file. Only one chunk can be written with this function.
`filters` is a mask providing a record of which filters are used with the chunk. The default value of the mask is zero (0), indicating that all enabled filters are applied. A filter is skipped if the bit corresponding to the filter's position in the pipeline (0 ≤ position < 32) is turned on. This mask is saved with the chunk in the file.
`offset` is an array specifying the logical position of the first element of the chunk in the dataset's dataspace. The length of the offset array must equal the number of dimensions, or rank, of the dataspace. The values in offset must not exceed the dimension limits and must specify a point that falls on a dataset chunk boundary.
`data_size` is the size in bytes of the chunk, representing the number of bytes to be read from the buffer `buf`. If the data chunk has been precompressed, `data_size` should be the size of the compressed data.
`buf` is the memory buffer containing data to be written to the chunk in the file. 

Attention
    Exercise caution when using [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file."), as they read and write data chunks directly in a file. [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") bypasses hyperslab selection, the conversion of data from one datatype to another, and the filter pipeline to write the chunk. Developers should have experience with these processes before using this function. 

Note
     [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") are currently not supported with parallel HDF5 and do not support variable-length types. 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8)H5Dwrite_multi()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dwrite_multi  | ( | size_t |  _count_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const void * |  _buf_[] )  
Writes raw data from a set buffers to a set of datasets. 
* * * 

Parameters
     [in] | count | Number of datasets to write to   
---|---|---  
[in] | dset_id | Identifiers of the datasets to write to   
[in] | mem_type_id | Identifiers of the memory datatypes   
[in] | mem_space_id | Identifiers of the memory dataspaces   
[in] | file_space_id | Identifiers of the datasets' dataspaces in the file   
[in] | dxpl_id | Identifier of a transfer property list   
[in] | buf | Buffers with data to be written to the file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Dwrite_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8 "Writes raw data from a set buffers to a set of datasets.") writes data to `count` datasets, whose identifiers are listed in the `dset_id` array, from multiple application memory buffers listed in the `buf` array. Data transfer properties are defined by the argument `dxpl_id`. The memory datatypes of each dataset are listed by identifier in the `mem_type_id` array. The parts of each dataset to write are listed by identifier in the `file_space_id` array, and the parts of each application memory buffer to write from are listed by identifier in the `mem_space_id` array. All array parameters have length `count`.
This function will produce the same results as `count` calls to [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."). Information listed in that function's documentation about the specifics of its behaviour also apply to [H5Dwrite_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8 "Writes raw data from a set buffers to a set of datasets."). By calling [H5Dwrite_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8 "Writes raw data from a set buffers to a set of datasets.") instead of multiple calls to [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."), however, the library can in some cases pass information about the entire I/O operation to the file driver, which can improve performance.
All datasets must be in the same HDF5 file, and each unique dataset may only be listed once. If this function is called collectively in parallel, each rank must pass exactly the same list of datasets in `dset_id` , though the other parameters may differ. 

Since
    1.14.0 

See also
     [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.")
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


