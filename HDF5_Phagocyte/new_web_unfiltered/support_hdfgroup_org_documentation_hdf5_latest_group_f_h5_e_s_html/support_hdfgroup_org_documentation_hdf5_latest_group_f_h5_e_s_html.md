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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title0 "H2")
  * [ Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title1 "H2")
  * [ Variables](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title2 "H2")
  * [Function/Subroutine Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title3 "H2")
  * [◆ h5escancel_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title4 "H2")
  * [◆ h5esclose_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title5 "H2")
  * [◆ h5escreate_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title6 "H2")
  * [◆ h5esget_count_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title7 "H2")
  * [◆ h5esget_err_count_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title8 "H2")
  * [◆ h5esget_err_status_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title9 "H2")
  * [◆ h5esget_op_counter_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title10 "H2")
  * [◆ h5eswait_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title11 "H2")
  * [Variable Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title12 "H2")
  * [◆ h5es_none_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title13 "H2")
  * [◆ h5es_status_canceled_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title14 "H2")
  * [◆ h5es_status_fail_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title15 "H2")
  * [◆ h5es_status_in_progress_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title16 "H2")
  * [◆ h5es_status_succeed_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title17 "H2")
  * [◆ h5es_wait_forever_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title18 "H2")
  * [◆ h5es_wait_none_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#title19 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#func-members) | [Variables](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#var-members)
Fortran Event Set (H5ES) Interface
## Detailed Description 

See also
     [Event Set Interface (H5ES)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html "Event Set Interface."), C-API      [HDF5 Event Set](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e_s__u_g.html), User Guide 
##  Functions/Subroutines  
---  
subroutine  |  [h5escancel_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gaa43b32d71f2af278fc3b53f3be9bec21) (es_id, num_not_canceled, err_occurred, hdferr)  
| Attempt to cancel operations in an event set.   
  
subroutine  |  [h5esclose_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga8350535ea15b783459ce4c82d6a2df3e) (es_id, hdferr)  
| Terminates access to an event set.   
  
subroutine  |  [h5escreate_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2f0fc8458c431089e39f5e63f5ac2653) (es_id, hdferr)  
| Creates an event set.   
  
subroutine  |  [h5esget_count_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga58ed9668caf37f0e21e171705f69863b) (es_id, count, hdferr)  
| Retrieves number of events in an event set.   
  
subroutine  |  [h5esget_err_count_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2c0d4949c2ec09174bfaa46404db9d2c) (es_id, num_errs, hdferr)  
| Retrieves the number of failed operations.   
  
subroutine  |  [h5esget_err_status_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga58c019c10880b61fbe3af942b3be315d) (es_id, err_occurred, hdferr)  
| Checks for failed operations.   
  
subroutine  |  [h5esget_op_counter_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga6a47cfa2910b95cc195461b2e3873101) (es_id, counter, hdferr)  
| Retrieves the next operation counter to be assigned in an event set.   
  
subroutine  |  [h5eswait_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2c15e17e8d08406431375dba09ea7453) (es_id, timeout, num_in_progress, err_occurred, hdferr)  
| Waits for operations in event set to complete.   
  
##  Variables  
---  
integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943))  | [h5es_none_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gad2d91c49747e1e49d87b931db19da7be)  
| H5ES_NONE.   
  
integer  | [h5es_status_canceled_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gac4e50e7771ec52e16e042eaf92a3e446)  
| H5ES_STATUS_CANCELED.   
  
integer  | [h5es_status_fail_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga55c8630628cf58ef10c718583200170a)  
| H5ES_STATUS_FAIL.   
  
integer  | [h5es_status_in_progress_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga9e1525d9a6649719d46b9ff938aa8750)  
| H5ES_STATUS_IN_PROGRESS.   
  
integer  | [h5es_status_succeed_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gaf7d7e05deb0b01d80166da1fe1998b14)  
| H5ES_STATUS_SUCCEED.   
  
integer(c_int64_t)  | [h5es_wait_forever_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gab5c96ecc63ca17f47eaaca615c68f16f)  
| H5ES_WAIT_FOREVER.   
  
integer(c_int64_t)  | [h5es_wait_none_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga0846c695aadb2ee930e0b2170bd58b62)  
| H5ES_WAIT_NONE.   
  
## Function/Subroutine Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gaa43b32d71f2af278fc3b53f3be9bec21)h5escancel_f()
subroutine h5escancel_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(out) |  _num_not_canceled_ ,   
|  | logical, intent(out) |  _err_occurred_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Attempt to cancel operations in an event set.  

Parameters
     es_id | Event set identifier   
---|---  
num_not_canceled | The number of events not canceled   
err_occurred | Status indicating if error is present in the event set   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5EScancel()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga31adb1cd449abdf71584af89195bf18e)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga8350535ea15b783459ce4c82d6a2df3e)h5esclose_f()
subroutine h5esclose_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer, intent(out) |  _hdferr_ )  
Terminates access to an event set.  

Parameters
     es_id | Event set identifier   
---|---  
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1159e2c6748f200857dffa55011ae060)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2f0fc8458c431089e39f5e63f5ac2653)h5escreate_f()
subroutine h5escreate_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(out) |  _es_id_ ,   
---|---|---|---  
|  | integer, intent(out) |  _hdferr_ )  
Creates an event set.  

Parameters
     es_id | Event set identifier   
---|---  
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5EScreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#gae3cd0a94586acf2d824033ef59fd3ccc)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga58ed9668caf37f0e21e171705f69863b)h5esget_count_f()
subroutine h5esget_count_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(out) |  _count_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Retrieves number of events in an event set.  

Parameters
     es_id | Event set identifier   
---|---  
count | The number of events in the event set   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESget_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1a5a9212958bf7054a56107587d480d2)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2c0d4949c2ec09174bfaa46404db9d2c)h5esget_err_count_f()
subroutine h5esget_err_count_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(out) |  _num_errs_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Retrieves the number of failed operations.  

Parameters
     es_id | Event set identifier   
---|---  
num_errs | Number of errors   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESget_err_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga42b502d05e0dff40ec550c85bb54ca1c)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga58c019c10880b61fbe3af942b3be315d)h5esget_err_status_f()
subroutine h5esget_err_status_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | logical, intent(out) |  _err_occurred_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Checks for failed operations.  

Parameters
     es_id | Event set identifier   
---|---  
err_occurred | Status indicating if error is present in the event set   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESget_err_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga3f1d63eb93c8466e0132303f506f6d33)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga6a47cfa2910b95cc195461b2e3873101)h5esget_op_counter_f()
subroutine h5esget_op_counter_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer(c_int64_t), intent(out) |  _counter_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Retrieves the next operation counter to be assigned in an event set.  

Parameters
     es_id | Event set identifier   
---|---  
counter | The number of events in the event set   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESget_op_counter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga1880fb2353f677ef56a6204706cec4d0)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga2c15e17e8d08406431375dba09ea7453)h5eswait_f()
subroutine h5eswait_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _es_id_ ,   
---|---|---|---  
|  | integer(c_int64_t), intent(in) |  _timeout_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(out) |  _num_in_progress_ ,   
|  | logical, intent(out) |  _err_occurred_ ,   
|  | integer, intent(out) |  _hdferr_ )  
Waits for operations in event set to complete.  

Parameters
     es_id | Event set identifier   
---|---  
timeout | The number of events in the event set   
num_in_progress | The number of operations still in progress   
err_occurred | Flag if an operation in the event set failed   
hdferr | Returns 0 if successful and -1 if it fails.  
See C API: [H5ESwait()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e_s.html#ga4f125d5ee5e59e3f6005f07723d306aa)
## Variable Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gad2d91c49747e1e49d87b931db19da7be)h5es_none_f
integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) h5es_none_f  
---  
H5ES_NONE. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gac4e50e7771ec52e16e042eaf92a3e446)h5es_status_canceled_f
integer h5es_status_canceled_f  
---  
H5ES_STATUS_CANCELED. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga55c8630628cf58ef10c718583200170a)h5es_status_fail_f
integer h5es_status_fail_f  
---  
H5ES_STATUS_FAIL. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga9e1525d9a6649719d46b9ff938aa8750)h5es_status_in_progress_f
integer h5es_status_in_progress_f  
---  
H5ES_STATUS_IN_PROGRESS. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gaf7d7e05deb0b01d80166da1fe1998b14)h5es_status_succeed_f
integer h5es_status_succeed_f  
---  
H5ES_STATUS_SUCCEED. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#gab5c96ecc63ca17f47eaaca615c68f16f)h5es_wait_forever_f
integer(c_int64_t) h5es_wait_forever_f  
---  
H5ES_WAIT_FOREVER. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_e_s.html#ga0846c695aadb2ee930e0b2170bd58b62)h5es_wait_none_f
integer(c_int64_t) h5es_wait_none_f  
---  
H5ES_WAIT_NONE. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


