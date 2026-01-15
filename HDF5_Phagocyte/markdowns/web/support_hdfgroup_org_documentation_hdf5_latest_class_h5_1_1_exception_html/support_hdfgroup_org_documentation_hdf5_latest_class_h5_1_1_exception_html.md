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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title0 "H2")
  * [ Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title1 "H2")
  * [ Static Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title2 "H2")
  * [ Static Protected Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title3 "H2")
  * [Constructor & Destructor Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title4 "H2")
  * [◆ Exception() [1/3]](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title5 "H2")
  * [◆ Exception() [2/3]](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title6 "H2")
  * [◆ Exception() [3/3]](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title7 "H2")
  * [◆ ~Exception()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title8 "H2")
  * [Member Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title9 "H2")
  * [◆ clearErrorStack()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title10 "H2")
  * [◆ dontPrint()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title11 "H2")
  * [◆ getAutoPrint()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title12 "H2")
  * [◆ getCDetailMsg()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title13 "H2")
  * [◆ getCFuncName()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title14 "H2")
  * [◆ getDetailMsg()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title15 "H2")
  * [◆ getFuncName()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title16 "H2")
  * [◆ getMajorString()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title17 "H2")
  * [◆ getMinorString()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title18 "H2")
  * [◆ printErrorStack()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title19 "H2")
  * [◆ setAutoPrint()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title20 "H2")
  * [◆ walkErrorStack()](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title21 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title22 "H2")
  * [◆ DEFAULT_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#title23 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#pub-methods) | [Static Public Member Functions](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#pub-static-methods) | [Static Protected Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#pro-static-attribs)
Exception Class Reference
`#include <c++/src/H5Exception.h>`
## Detailed Description
[Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html "Exception provides wrappers of HDF5 error handling functions.") provides wrappers of HDF5 error handling functions. 
Many classes are derived from [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html "Exception provides wrappers of HDF5 error handling functions.") for specific HDF5 C interfaces. 
![+](https://support.hdfgroup.org/documentation/hdf5/latest/closed.png) Inheritance diagram for Exception:
![](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.png)
##  Public Member Functions  
---  
|  [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#abfbc23b99b2e78b609d50ac688611236) ()  
| Default constructor.   
  
|  [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#acad34d4fc0c2bfc6c8739c3db96d317e) (const [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) &orig)  
| Copy constructor: same HDF5 object as _original_.   
  
|  [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a480eadecaf7b936ad9aebab914243a76) (const std::string &func_name, const std::string &message=[DEFAULT_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac6cad9847003e45256651f63be151495))  
| Creates an exception with the name of the function, in which the failure occurs, and an optional detailed message.   
  
const char *  |  [getCDetailMsg](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a9261d8c971d2221064de465a044e23a4) () const  
| Returns the detailed message set at the time the exception is thrown.   
  
const char *  |  [getCFuncName](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a0687e8cfdd6010cf1f7389acec979d8f) () const  
| Returns the name of the function, where the exception is thrown.   
  
std::string  |  [getDetailMsg](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ad8da747cc2259a2f6691b41bd0f19248) () const  
| Returns the detailed message set at the time the exception is thrown.   
  
std::string  |  [getFuncName](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac4425f35e8f44fc664a5c6d505d233d5) () const  
| Returns the name of the function, where the exception is thrown.   
  
std::string  |  [getMajorString](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a8b5150b7308060a67989c4fe1aab7acf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) err_major_id) const  
| Returns a text string that describes the error specified by a major error number.   
  
std::string  |  [getMinorString](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a1797c734da505d69bfbf3651f0da570f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) err_minor_id) const  
| Returns a text string that describes the error specified by a minor error number.   
  
virtual  |  [~Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a1de477f66297b28e1bb70ca104523d26) ()=default  
##  Static Public Member Functions  
---  
static void  |  [clearErrorStack](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a29f3b25b1b290a2c867a4015a8bb808c) ()  
| Clears the error stack for the current thread.   
  
static void  |  [dontPrint](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a7deb933f2a3bd479bc592862b90428ed) ()  
| Turns off the automatic error printing from the C library.   
  
static void  |  [getAutoPrint](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#aca573fd364e64a80f9eb9cdd0dcf9591) ([H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca) &func, void **client_data)  
| Retrieves the current settings for the automatic error stack traversal function and its data.   
  
static void  |  [printErrorStack](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a069c7489dd4f423fde47a65b8d25b23d) (FILE *stream=stderr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) err_stack=[H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455))  
| Prints the error stack in a default manner.   
  
static void  |  [setAutoPrint](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#af2fb84ab9840878fbc07812a6e1a930e) ([H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca) &func, void *client_data)  
| Turns on the automatic error printing.   
  
static void  |  [walkErrorStack](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#afb3a720038672851192842936a120721) ([H5E_direction_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813) direction, [H5E_walk2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aa0fc6445c613e4159a17d28ca61be825) func, void *client_data)  
| Walks the error stack for the current thread, calling the specified function.   
  
##  Static Protected Attributes  
---  
static const char  |  [DEFAULT_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac6cad9847003e45256651f63be151495) [] = "No detailed information provided"  
## Constructor & Destructor Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a480eadecaf7b936ad9aebab914243a76)Exception() [1/3]
[Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) | ( | const std::string & |  _func_ ,   
---|---|---|---  
|  | const std::string & |  _message_ = [DEFAULT_MSG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac6cad9847003e45256651f63be151495) )  
Creates an exception with the name of the function, in which the failure occurs, and an optional detailed message.  

Parameters
     func | - IN: Name of the function where failure occurs   
---|---  
message | - IN: Message on the failure   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#abfbc23b99b2e78b609d50ac688611236)Exception() [2/3]
[Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) | ( |  | ) |   
---|---|---|---|---  
Default constructor. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#acad34d4fc0c2bfc6c8739c3db96d317e)Exception() [3/3]
[Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) | ( | const [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) & | _orig_ | ) |   
---|---|---|---|---|---  
Copy constructor: same HDF5 object as _original_.  

Parameters
     orig | - IN: [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html "Exception provides wrappers of HDF5 error handling functions.") instance to copy   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a1de477f66297b28e1bb70ca104523d26)~Exception()
| virtual ~[Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html) | ( |  | ) |   
---|---|---|---|---  
virtualdefault  
## Member Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a29f3b25b1b290a2c867a4015a8bb808c)clearErrorStack()
| void clearErrorStack  | ( |  | ) |   
---|---|---|---|---  
static  
Clears the error stack for the current thread.  

Description
    The stack is also cleared whenever a C API function is called, with certain exceptions (for instance, `H5Eprint`). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a7deb933f2a3bd479bc592862b90428ed)dontPrint()
| void dontPrint  | ( |  | ) |   
---|---|---|---|---  
static  
Turns off the automatic error printing from the C library. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#aca573fd364e64a80f9eb9cdd0dcf9591)getAutoPrint()
| void getAutoPrint  | ( |  [H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca) & |  _func_ ,   
---|---|---|---  
|  | void ** |  _client_data_ )  
static  
Retrieves the current settings for the automatic error stack traversal function and its data.  

Parameters
     func | - OUT: Current setting for the function to be called upon an error condition   
---|---  
client_data | - OUT: Current setting for the data passed to the error function   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a9261d8c971d2221064de465a044e23a4)getCDetailMsg()
const char * getCDetailMsg  | ( |  | ) |  const  
---|---|---|---|---  
Returns the detailed message set at the time the exception is thrown.  

Returns
    Text message - `char` pointer 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a0687e8cfdd6010cf1f7389acec979d8f)getCFuncName()
const char * getCFuncName  | ( |  | ) |  const  
---|---|---|---|---  
Returns the name of the function, where the exception is thrown.  

Returns
    Text message - `char` pointer 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ad8da747cc2259a2f6691b41bd0f19248)getDetailMsg()
std::string getDetailMsg  | ( |  | ) |  const  
---|---|---|---|---  
Returns the detailed message set at the time the exception is thrown.  

Returns
    Text message - `H5std_string`
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac4425f35e8f44fc664a5c6d505d233d5)getFuncName()
std::string getFuncName  | ( |  | ) |  const  
---|---|---|---|---  
Returns the name of the function, where the exception is thrown.  

Returns
    Text message - `H5std_string`
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a8b5150b7308060a67989c4fe1aab7acf)getMajorString()
std::string getMajorString  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _err_major_ | ) |  const  
---|---|---|---|---|---  
Returns a text string that describes the error specified by a major error number.  

Parameters
     err_major | - IN: Major error number   
---|--- 

Returns
    Major error string  

Description
    In the failure case, the string "Invalid major error number" will be returned.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a1797c734da505d69bfbf3651f0da570f)getMinorString()
std::string getMinorString  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _err_minor_ | ) |  const  
---|---|---|---|---|---  
Returns a text string that describes the error specified by a minor error number.  

Parameters
     err_minor | - IN: Minor error number   
---|--- 

Returns
    Minor error string  

Description
    In the failure case, the string "Invalid minor error number" will be returned.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#a069c7489dd4f423fde47a65b8d25b23d)printErrorStack()
| void printErrorStack  | ( | FILE * |  _stream_ = stderr,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _err_stack_ = [H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455) )  
static  
Prints the error stack in a default manner.  

Parameters
     stream | - IN: File pointer, default to stderr   
---|---  
err_stack | - IN: Error stack ID, default to [H5E_DEFAULT(0)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455)  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#af2fb84ab9840878fbc07812a6e1a930e)setAutoPrint()
| void setAutoPrint  | ( |  [H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca) & |  _func_ ,   
---|---|---|---  
|  | void * |  _client_data_ )  
static  
Turns on the automatic error printing.  

Parameters
     func | - IN: Function to be called upon an error condition   
---|---  
client_data | - IN: Data passed to the error function  

Description
    When the library is first initialized the auto printing function, _func_ , is set to the C API `H5Eprint` and _client_data_ is the standard error stream pointer, `stderr`. Automatic stack traversal is always in the `H5E_WALK_DOWNWARD` direction.      Users are encouraged to write their own more specific error handlers   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#afb3a720038672851192842936a120721)walkErrorStack()
| void walkErrorStack  | ( | [H5E_direction_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813) |  _direction_ ,   
---|---|---|---  
|  | [H5E_walk2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aa0fc6445c613e4159a17d28ca61be825) |  _func_ ,   
|  | void * |  _client_data_ )  
static  
Walks the error stack for the current thread, calling the specified function.  

Parameters
     direction | - IN: Direction in which the error stack is to be walked   
---|---  
func | - IN: Function to be called for each error encountered   
client_data | - IN: Data passed to the error function  

Description
    Valid values for _direction_ include:   
  * `H5E_WALK_UPWARD` - begin with the most specific error and end at the API 
  * `H5E_WALK_DOWNWARD` - begin at the API and end at the inner-most function where the error was first detected 

    The function specified by _func_ will be called for each error in the error stack. The `H5E_walk_t` prototype is as follows: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e))(int n, [H5E_error_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2592cae9c5fa61a6045b621cd94d3383) *err_desc, void *client_data)
int n - Indexed position of the error in the stack; it begins at zero
regardless of stack traversal direction
[H5E_error_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2592cae9c5fa61a6045b621cd94d3383) *err_desc - Pointer to a data structure describing the
error. This structure is listed below.
void *client_data - Pointer to client data in the format expected by
the user-defined function.
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5E_error_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a2592cae9c5fa61a6045b621cd94d3383)
#define H5E_error_t
**Definition** H5version.h:1213
[H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e)
#define H5E_walk_t
**Definition** H5version.h:1214     Data structure to describe the error: 
typedef struct [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html) {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cls_id; //class ID
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) maj_num; //major error ID
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) min_num; //minor error number
const char *func_name; //function in which error occurred
const char *file_name; //file in which error occurred
unsigned line; //line in file where error occurs
const char *desc; //optional supplied description
} [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html)
**Definition** H5Epublic.h:35
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#ac6cad9847003e45256651f63be151495)DEFAULT_MSG
| const char DEFAULT_MSG = "No detailed information provided"  
---  
staticprotected  
* * *
The documentation for this class was generated from the following files:
  * c++/src/[H5Exception.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_exception_8h.html)
  * c++/src/[H5Exception.cpp](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_exception_8cpp.html)


  * [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)
  * [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


