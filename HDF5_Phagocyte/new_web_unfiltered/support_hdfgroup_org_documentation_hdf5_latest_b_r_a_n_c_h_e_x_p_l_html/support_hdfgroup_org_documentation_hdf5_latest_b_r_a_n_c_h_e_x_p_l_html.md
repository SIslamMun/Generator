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
  * [ develop branch](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html#title0 "H1")
  * [ Maintenance branches](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html#title1 "H1")
  * [ Release branches](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html#title2 "H1")
  * [ feature/*](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html#title3 "H1")
  * [ inactive/*](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html#title4 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Git Branching Model Explained
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
This document describes current HDF5 branches.
Branches are tested nightly and testing results are available at <https://my.cdash.org/index.php?project=HDF5>. Commits that break daily testing should be fixed by 3:00 pm Central time or reverted.   
We encourage code contributors to check the status of their commits. If you have any questions, please contact [help@.nosp@m.hdfg.nosp@m.roup..nosp@m.org](https://support.hdfgroup.org/documentation/hdf5/latest/_b_r_a_n_c_h_e_x_p_l.html).
#  develop branch
Develop is the main branch whose source code always reflects a state with the latest delivered development changes for the next major release of HDF5. This is also considered the integration branch, as **all** new features are integrated into this branch from respective feature branches. Although develop is considered an integration branch, it is not an unstable branch. All code merged to develop is expected to pass all GitHub actions and daily tests.
#  Maintenance branches
Each currently supported release line of HDF5 (e.g. 1.8.x, 1.10.x, 1.12.x, 1.14.x) has an associated branch with the name hdf5_1_10, etc.. Maintenance branches are similar to the develop branch, except the source code in a maintenance branch always reflects a state with the latest delivered development changes for the next **maintenance** release of that particular supported release-line of HDF5. **Some** new features will be integrated into a release maintenance branch, depending on whether or not those features can be introduced in minor releases. Maintenance branches are removed when a release-line is retired from support.
#  Release branches
Release branches are used to prepare a new production release. They are primarily used to allow for last minute dotting of i's and crossing of t's (things like setting the release version, finalizing release notes, and generating files) and do not include new development. They are created from the maintenance branch at the time of the maintenance release and have names like hdf5_1_10_N, where N is the minor release number. Once the release is done it is tagged, with a slightly different format: hdf5-1_10_N. Release branches are deleted after the tag has been created. If we have to create a patch version of a release (which is rare), we create a branch off of the tag.
#  feature/*
Feature branches are temporary branches used to develop new features in HDF5. Feature branches branch off of develop and exist as long as the feature is under development. When the feature is complete, the branch is merged back into develop, as well as into any support branches in which the change will be included, and then the feature branch is removed.
Ideally, all feature branches should contain a BRANCH.md file in the root directory that explains the purpose of the branch, contact information for the person responsible, and, if possible, some clues about the branch's life cycle (so we have an idea about when it can be deleted, merged, or declared inactive).
Minor bug fixes and refactoring work usually takes place on personal forks, not feature branches.
#  inactive/*
These branches are for experimental features that were developed in the past, have not been merged to develop, and are not under active development. The exception to this is that some feature branches are labeled inactive and preserved for a short time after merging to develop. Integration branches are usually not kept in sync with the develop branch.
As for feature branches, inactive branches should have a BRANCH.md file as described above.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


