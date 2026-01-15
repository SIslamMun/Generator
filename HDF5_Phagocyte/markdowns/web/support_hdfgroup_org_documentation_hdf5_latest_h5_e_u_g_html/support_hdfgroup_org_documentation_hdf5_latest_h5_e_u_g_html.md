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
  * [ HDF5 Error Handling](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title1 "H2")
  * [ Error Handling Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title2 "H2")
  * [ Programming Model for Error Handling](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title3 "H2")
  * [ Basic Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title4 "H2")
  * [ Advanced Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Error Handling
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Error Handling
The HDF5 library provides an error reporting mechanism for both the library itself and for user application programs. It can trace errors through function stack and error information like file name, function name, line number, and error description.
##  Introduction
The HDF5 Library provides an error reporting mechanism for both the library itself and for user application programs. It can trace errors through function stack and error information like file name, function name, line number, and error description.
[Basic Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_ops) discusses the basic error concepts such as error stack, error record, and error message and describes the related API functions. These concepts and functions are sufficient for application programs to trace errors inside the HDF5 Library.
[Advanced Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_adv) talks about the advanced concepts of error class and error stack handle and talks about the related functions. With these concepts and functions, an application library or program using the HDF5 Library can have its own error report blended with HDF5's error report.
Starting with Release 1.8, we have a new set of Error Handling API functions. For the purpose of backward compatibility with version 1.6 and before, we still keep the old API functions, [H5Epush1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga7e2d223ad3bf68fe35f343b97edf0e92), [H5Eprint1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga9c71eb8e5b7261668e2e8926f1822365), [H5Ewalk1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga8d54a0f62f9ea625bdeab8e5e0c894c4), [H5Eclear1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0f2ee26cbe35c5dde49d615fc31ea2f6), [H5Eget_auto1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0ca4dd7ed560882a7da176a0e2325707), [H5Eset_auto1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gab7e1c2db4a0811b70227833b5462eea8). These functions do not have the error stack as a parameter. The library allows them to operate on the default error stack. (The H5E compatibility macros will choose the correct function based on the parameters)
The old API is similar to functionality discussed in [Basic Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_ops). The functionality discussed in [Advanced Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_adv),the ability of allowing applications to add their own error records, is the new design for the Error Handling API.
##  Error Handling Function Summaries
see [Error Handling (H5E)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html) reference manual
##  Programming Model for Error Handling
The HDF5 error handling programming model provides a structured approach for detecting, reporting, and managing errors in applications. The model is based on error stacks that accumulate contextual information as errors propagate through the library call stack.
###  Basic Programming Model
Applications typically use one of two approaches for error handling:
  * **Automatic Error Handling** : By default, HDF5 automatically prints error information to stderr when errors occur. This is suitable for development and debugging but may not be appropriate for production applications.


  * **Custom Error Handling** : Applications can register custom error handling functions using [H5Eset_auto](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga245383d63fba5c4282ce8e9ef8488702) to gain full control over error reporting and recovery. Custom handlers can log errors, display messages to users, or implement application-specific recovery strategies.


###  Error Detection
HDF5 functions indicate errors through their return values: 
  * Functions returning `hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)` (identifiers) return a negative value on error 
  * Functions returning `herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)` (error codes) return a negative value on error 
  * Functions returning pointers return NULL on error 
  * Functions returning `htri_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca)` (tri-state) return negative on error, zero for false, positive for true


Applications should check return values from all HDF5 API calls to detect errors.
###  Error Handling Strategies
Common error handling strategies include:
  * **Print and Continue** : Use default error handling to print diagnostic information and continue execution where possible. This is the default behavior.


  * **Silent Error Handling** : Disable automatic error printing with [H5Eset_auto(H5E_DEFAULT, NULL, NULL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga245383d63fba5c4282ce8e9ef8488702) and check return values explicitly. This is common in production code.


  * **Custom Error Handler** : Install a custom error handler to implement application-specific error logging, user notification, or recovery procedures.


  * **Error Stack Inspection** : Walk the error stack with [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0) to examine detailed error information and make programmatic decisions based on specific error conditions.



See also
     [Basic Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_ops) for details on error stack operations and [Advanced Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_adv) for advanced error handling features including custom error classes and messages.
##  Basic Error Handling Operations
Let us first try to understand the error stack. An error stack is a collection of error records. Error records can be pushed onto or popped off the error stack. By default, when an error occurs deep within the HDF5 Library, an error record is pushed onto an error stack and that function returns a failure indication. Its caller detects the failure, pushes another record onto the stack, and returns a failure indication. This continues until the API function called by the application returns a failure indication. The next API function being called will reset the error stack. All HDF5 Library error records belong to the same error class. For more information, see [Advanced Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_adv).
###  Error Stack and Error Message
In normal circumstances, an error causes the stack to be printed on the standard error stream automatically. This automatic error stack is the library's default stack. For all the functions in this section, whenever an error stack ID is needed as a parameter, [H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455) can be used to indicate the library's default stack. The first error record of the error stack, number #000, is produced by the API function itself and is usually sufficient to indicate to the application what went wrong.
If an application calls [H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0) on a predefined datatype then the following message is printed on the standard error stream. This is a simple error that has only one component, the API function; other errors may have many components.
_An Error Message Example_
HDF5-DIAG: Error detected in HDF5 (1.10.9) thread 0.
#000: H5T.c line ### in H5Tclose(): predefined datatype
major: Function argument
minor: Bad value
In the example above, we can see that an error record has a major message and a minor message. A major message generally indicates where the error happens. The location can be a dataset or a dataspace, for example. A minor message explains further details of the error. An example is “unable to open file”. Another specific detail about the error can be found at the end of the first line of each error record. This error description is usually added by the library designer to tell what exactly goes wrong. In the example above, the “predefined datatype” is an error description.
###  Print and Clear an Error Stack
Besides the automatic error report, the error stack can also be printed and cleared by the functions [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56) and [H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2). If an application wishes to make explicit calls to [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56) to print the error stack, the automatic printing should be turned off to prevent error messages from being displayed twice (see [H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)).
_To print an error stack:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack, FILE * stream)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56)
herr_t H5Eprint2(hid_t err_stack, FILE *stream)
Prints the specified error stack in a default manner.
This function prints the error stack specified by error_stack on the specified stream, stream. If the error stack is empty, a one‐line message will be printed. The following is an example of such a message. This message would be generated if the error was in the HDF5 Library. 
HDF5-DIAG: Error detected in HDF5 Library version: 1.10.9 thread 0.
_To clear an error stack:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack)
[H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2)
herr_t H5Eclear2(hid_t err_stack)
Clears the specified error stack or the error stack for the current thread.
The [H5Eclear2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2) function shown above clears the error stack specified by error_stack. [H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455) can be passed in to clear the current error stack. The current stack is also cleared whenever an API function is called; there are certain exceptions to this rule such as [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56).
###  Mute Error Stack
Sometimes an application calls a function for the sake of its return value, fully expecting the function to fail; sometimes the application wants to call [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56) explicitly. In these situations, it would be misleading if an error message were still automatically printed. Using the [H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69) function can control the automatic printing of error messages.
_To enable or disable automatic printing of errors:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack, [H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051) func, void *client_data)
[H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051)
#define H5E_auto_t
**Definition** H5version.h:1563
[H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)
herr_t H5Eset_auto2(hid_t estack_id, H5E_auto2_t func, void *client_data)
Turns automatic error printing on or off.
The [H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69) function can be used to turn on or off the automatic printing of errors for the error stack specified by error_stack. When turned on (non‐null func pointer), any API function which returns an error indication will first call func, passing it client_data as an argument. When the library is first initialized the auto printing function is set to [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56) and client_data is the standard error stream pointer, stderr.
_To see the current settings:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eget_auto](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa088620ef0b6c4ac2abcf57d61c8cdb8)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack, [H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051) * func, void **client_data)
[H5Eget_auto](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa088620ef0b6c4ac2abcf57d61c8cdb8)
#define H5Eget_auto
**Definition** H5version.h:1168
The function above returns the current settings for the automatic error stack traversal function, func, and its data, client_data. If either or both of the arguments are null, then the value is not returned.
An application can temporarily turn off error messages while “probing” a function. See the example below.
_Example: Turn off error messages while probing a function_
// Save old error handler
[H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca) oldfunc;
void *old_client_data;
[H5Eget_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga2eda33cbadd9be5bfddbaa91e863c936)(error_stack, &old_func, &old_client_data);
// Turn off error handling
[H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)(error_stack, NULL, NULL);
// Probe. Likely to fail, but that's okay
status = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) (......);
// Restore previous error handler
[H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)(error_stack, old_func, old_client_data);
[H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca)
herr_t(* H5E_auto2_t)(hid_t estack, void *client_data)
Callback function for H5Eset_auto2()
**Definition** H5Epublic.h:191
[H5Eget_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga2eda33cbadd9be5bfddbaa91e863c936)
herr_t H5Eget_auto2(hid_t estack_id, H5E_auto2_t *func, void **client_data)
Returns the settings for the automatic error stack traversal function and its data.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
Or automatic printing can be disabled altogether and error messages can be explicitly printed.
_Example: Disable automatic printing and explicitly print error messages_
// Turn off error handling permanently
[H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)(error_stack, NULL, NULL);
// If failure, print error message
if ([H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) (....)<0) {
[H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56)([H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455), stderr);
exit (1);
}
[H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455)
#define H5E_DEFAULT
**Definition** H5Epublic.h:25
###  Customized Printing of an Error Stack
Applications are allowed to define an automatic error traversal function other than the default [H5Eprint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa7e93cb96b399e5853721258872435a8). For instance, one can define a function that prints a simple, one‐line error message to the standard error stream and then exits. The first example below defines a such a function. The second example below installs the function as the error handler.
_Example: Defining a function to print a simple error message_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
my_hdf5_error_handler(void *unused)
{
fprintf (stderr, “An HDF5 error was detected. Bye.\\\n”);
exit (1);
}
_Example: The user‐defined error handler_
[H5Eset_auto2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69)([H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455), my_hdf5_error_handler, NULL);
###  Walk through the Error Stack
The [H5Eprint2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56) function is actually just a wrapper around the more complex [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0) function which traverses an error stack and calls a user‐defined function for each member of the stack. The example below shows how [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0) is used. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) err_stack, [H5E_direction_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813) direction,
[H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e) func, void *client_data)
[H5E_direction_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813)
H5E_direction_t
**Definition** H5Epublic.h:153
[H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e)
#define H5E_walk_t
**Definition** H5version.h:1214
[H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0)
#define H5Ewalk
**Definition** H5version.h:1212
The error stack err_stack is traversed and func is called for each member of the stack. Its arguments are an integer sequence number beginning at zero (regardless of direction) and the client_data pointer. If direction is [H5E_WALK_UPWARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813abedac175e65559c32551f132a78e1861), then traversal begins at the inner‐most function that detected the error and concludes with the API function. Use [H5E_WALK_DOWNWARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ae30cff307b364e94ce2d552edbca6813ab0ba668f408b6c98dade64021973fb49) for the opposite order.
###  Traverse an Error Stack with a Callback Function
An error stack traversal callback function takes three arguments: n is a sequence number beginning at zero for each traversal, eptr is a pointer to an error stack member, and client_data is the same pointer used in the example above passed to [H5Ewalk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0). See the example below. 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5E_walk_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7bbdcafe87980d81a22ae5226f85c84e))(unsigned n, [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html) *eptr, void *client_data)
[H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html)
**Definition** H5Epublic.h:35
The [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html) structure is shown below. 
typedef struct {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cls_id;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) maj_num;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) min_num;
unsigned line;
const char *func_name;
const char *file_name;
const char *desc;
} [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html);
The maj_num and min_num are major and minor error IDs, func_name is the name of the function where the error was detected, file_name and line locate the error within the HDF5 Library source code, and desc points to a description of the error.
The following example shows a user‐defined callback function.
_A user‐defined callback function Example_
#define MSG_SIZE 64
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
custom_print_cb(unsigned n, const [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html) *err_desc, void *client_data)
{
FILE *stream = (FILE *)client_data;
char maj[MSG_SIZE];
char min[MSG_SIZE];
char cls[MSG_SIZE];
const int indent = 4;
// Get descriptions for the major and minor error numbers
if([H5Eget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf360b8e61728b421ee78438e4f57a001)(err_desc->[cls_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#aeec7f232ca3cf2df54f166d02bd4a1ab), cls, MSG_SIZE) < 0)
TEST_ERROR;
if([H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)(err_desc->[maj_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a3602dbcfe9ecf12271a04ae00de5a41d), NULL, maj, MSG_SIZE) < 0)
TEST_ERROR;
if([H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)(err_desc->[min_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#ab6bbaef82664e5deee0fb3c303a14bf6), NULL, min, MSG_SIZE) < 0)
TEST_ERROR;
fprintf (stream, “%*serror #%03d: %s in %s():
line %u\\\n”,
indent, “”, n, err_desc->[file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a89e7a505203c95130b3355a263a38fe2),
err_desc->[func_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a0ec1958bc64e47b7629ce65b76414ce0), err_desc->[line](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a05ef0c4dbeec4fc8ccb225de9c26d896));
fprintf (stream, “%*sclass: %s\\\n”, indent*2, “”, cls);
fprintf (stream, “%*smajor: %s\\\n”, indent*2, “”, maj);
fprintf (stream, “%*sminor: %s\\\n”, indent*2, “”, min);
return 0;
error:
return -1;
}
[H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)
ssize_t H5Eget_msg(hid_t msg_id, H5E_type_t *type, char *msg, size_t size)
Retrieves an error message.
[H5Eget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf360b8e61728b421ee78438e4f57a001)
ssize_t H5Eget_class_name(hid_t class_id, char *name, size_t size)
Retrieves error class name.
[H5E_error2_t::line](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a05ef0c4dbeec4fc8ccb225de9c26d896)
unsigned line
**Definition** H5Epublic.h:42
[H5E_error2_t::func_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a0ec1958bc64e47b7629ce65b76414ce0)
const char * func_name
**Definition** H5Epublic.h:44
[H5E_error2_t::maj_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a3602dbcfe9ecf12271a04ae00de5a41d)
hid_t maj_num
**Definition** H5Epublic.h:38
[H5E_error2_t::file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a89e7a505203c95130b3355a263a38fe2)
const char * file_name
**Definition** H5Epublic.h:46
[H5E_error2_t::min_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#ab6bbaef82664e5deee0fb3c303a14bf6)
hid_t min_num
**Definition** H5Epublic.h:40
[H5E_error2_t::cls_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#aeec7f232ca3cf2df54f166d02bd4a1ab)
hid_t cls_id
**Definition** H5Epublic.h:36
#### Programming Note for C++ Developers Using C Functions
If a C routine that takes a function pointer as an argument is called from within C++ code, the C routine should be returned from normally.
Examples of this kind of routine include callbacks such as [H5Pset_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2) and [H5Pset_type_conv_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga10a80b29444d933da1aa2003f46cf003) and functions such as [H5Tconvert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a) and [H5Ewalk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4ecc0f6a1ea5bb821373a5a7b8070655).
Exiting the routine in its normal fashion allows the HDF5 C Library to clean up its work properly. In other words, if the C++ application jumps out of the routine back to the C++ “catch” statement, the library is not given the opportunity to close any temporary data structures that were set up when the routine was called. The C++ application should save some state as the routine is started so that any problem that occurs might be diagnosed.
##  Advanced Error Handling Operations
The section above, see [Basic Error Handling Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#subsec_error_ops), discusses the basic error handling operations of the library. In that section, all the error records on the error stack are from the library itself. In this section, we are going to introduce the operations that allow an application program to push its own error records onto the error stack once it declares an error class of its own through the HDF5 Error API.
An error report shows both the library's error record and the application's error records. See the example below.
_An Error Report Example_
Error Test-DIAG: Error detected in Error Program (1.0)
thread 8192:
#000: ../../hdf5/test/error_test.c line ### in main():
Error test failed
major: Error in test
minor: Error in subroutine
#001: ../../hdf5/test/error_test.c line ### in
test_error(): [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) failed as supposed to
major: Error in IO
minor: Error in [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
HDF5-DIAG: Error detected in HDF5 (1.10.9) thread #####:
#002: ../../[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/src/H5Dio.c line ### in [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)():
not a dataset
major: Invalid arguments to routine
minor: Inappropriate type
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26
In the line above error record #002 in the example above, the starting phrase is HDF5. This is the error class name of the HDF5 Library. All of the library's error messages (major and minor) are in this default error class. The Error Test in the beginning of the line above error record #000 is the name of the application's error class. The first two error records, #000 and #001, are from application's error class. By definition, an error class is a group of major and minor error messages for a library (the HDF5 Library or an application library built on top of the HDF5 Library) or an application program. The error class can be registered for a library or program through the HDF5 Error API. Major and minor messages can be defined in an error class. An application will have object handles for the error class and for major and minor messages for further operation. See the example below.
_Example: The user‐defined error handler_
#define MSG_SIZE 64
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
custom_print_cb(unsigned n, const [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html) *err_desc,
void* client_data)
{
FILE *stream = (FILE *)client_data;
char maj[MSG_SIZE];
char min[MSG_SIZE];
char cls[MSG_SIZE];
const int indent = 4;
// Get descriptions for the major and minor error numbers
if([H5Eget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf360b8e61728b421ee78438e4f57a001)(err_desc->[cls_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#aeec7f232ca3cf2df54f166d02bd4a1ab), cls, MSG_SIZE) < 0)
TEST_ERROR;
if([H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)(err_desc->[maj_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a3602dbcfe9ecf12271a04ae00de5a41d), NULL, maj, MSG_SIZE) < 0)
TEST_ERROR;
if([H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)(err_desc->[min_num](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#ab6bbaef82664e5deee0fb3c303a14bf6), NULL, min, MSG_SIZE) < 0)
TEST_ERROR;
fprintf (stream, “%*serror #%03d: %s in %s():
line %u\\\n”,
indent, “”, n, err_desc->[file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a89e7a505203c95130b3355a263a38fe2),
err_desc->[func_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a0ec1958bc64e47b7629ce65b76414ce0), err_desc->[line](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html#a05ef0c4dbeec4fc8ccb225de9c26d896));
fprintf (stream, “%*sclass: %s\\\n”, indent*2, “”, cls);
fprintf (stream, “%*smajor: %s\\\n”, indent*2, “”, maj);
fprintf (stream, “%*sminor: %s\\\n”, indent*2, “”, min);
return 0;
error:
return -1;
}
###  More Error API Functions
The Error API has functions that can be used to register or unregister an error class, to create or close error messages, and to query an error class or error message. These functions are illustrated below.
_To register an error class:_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Eregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga70ecfb35ab871ddb1cd2dfa0ac5f740c)(const char* cls_name, const char* lib_name, const char* version)
[H5Eregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga70ecfb35ab871ddb1cd2dfa0ac5f740c)
hid_t H5Eregister_class(const char *cls_name, const char *lib_name, const char *version)
Registers a client library or application program to the HDF5 error API.
This function registers an error class with the HDF5 Library so that the application library or program can report errors together with the HDF5 Library.
_To add an error message to an error class:_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Ecreate_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga99a705d98873dcdd1bb6f9d5eebc5afd)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) class, [H5E_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3c) msg_type, const char* mesg)
[H5E_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3c)
H5E_type_t
**Definition** H5Epublic.h:30
[H5Ecreate_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga99a705d98873dcdd1bb6f9d5eebc5afd)
hid_t H5Ecreate_msg(hid_t cls, H5E_type_t msg_type, const char *msg)
Adds a major or minor error message to an error class.
This function adds an error message to an error class defined by an application library or program. The error message can be either major or minor which is indicated by parameter msg_type.
_To get the name of an error class:_
ssize_t [H5Eget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf360b8e61728b421ee78438e4f57a001)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) class_id, char* name, size_t size)
This function retrieves the name of the error class specified by the class ID.
_To retrieve an error message:_
ssize_t [H5Eget_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga64714effca13c23c4f95529256621fa0)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mesg_id, [H5E_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3c)* mesg_type, char* mesg, size_t size)
This function retrieves the error message including its length and type.
_To close an error message:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eclose_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga65b660eb16b25cc008585ba8990e8b15)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mesg_id)
[H5Eclose_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga65b660eb16b25cc008585ba8990e8b15)
herr_t H5Eclose_msg(hid_t err_id)
Closes an error message.
This function closes an error message.
_To remove an error class:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eunregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0383710d3984d19d3d2006d1151b88a2)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) class_id)
[H5Eunregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0383710d3984d19d3d2006d1151b88a2)
herr_t H5Eunregister_class(hid_t class_id)
Removes an error class.
This function removes an error class from the Error API.
The example below shows how an application creates an error class and error messages.
_Example: Create an error class and error messages_
// Create an error class
class_id = [H5Eregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga70ecfb35ab871ddb1cd2dfa0ac5f740c)(ERR_CLS_NAME, PROG_NAME, PROG_VERS);
// Retrieve class name
[H5Eget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf360b8e61728b421ee78438e4f57a001)(class_id, cls_name, cls_size);
// Create a major error message in the class
maj_id = [H5Ecreate_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga99a705d98873dcdd1bb6f9d5eebc5afd)(class_id, [H5E_MAJOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3cab0e1bfd4dab03f69f9a884e9079d3acc), “... ...”);
// Create a minor error message in the class
min_id = [H5Ecreate_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga99a705d98873dcdd1bb6f9d5eebc5afd)(class_id, [H5E_MINOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3caabc45a68c0f04871387677c099abb338), “... ...”);
[H5E_MINOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3caabc45a68c0f04871387677c099abb338)
@ H5E_MINOR
**Definition** H5Epublic.h:30
[H5E_MAJOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a415162f1ce9b31241977d0a5857caa3cab0e1bfd4dab03f69f9a884e9079d3acc)
@ H5E_MAJOR
**Definition** H5Epublic.h:30
The example below shows how an application closes error messages and unregisters the error class.
_Example: Closing error messages and unregistering the error class_
[H5Eclose_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga65b660eb16b25cc008585ba8990e8b15)(maj_id);
[H5Eclose_msg](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga65b660eb16b25cc008585ba8990e8b15)(min_id);
[H5Eunregister_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0383710d3984d19d3d2006d1151b88a2)(class_id);
###  Pushing an Application Error Message onto Error Stack
An application can push error records onto or pop error records off of the error stack just as the library does internally. An error stack can be registered, and an object handle can be returned to the application so that the application can manipulate a registered error stack.
_To register the current stack:_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Eget_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac66c0955a6d821a472a3a408cdc95ae6)(void)
[H5Eget_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac66c0955a6d821a472a3a408cdc95ae6)
hid_t H5Eget_current_stack(void)
Returns a copy of the current error stack.
This function registers the current error stack, returns an object handle, and clears the current error stack. An empty error stack will also be assigned an ID.
_To replace the current error stack with another:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eset_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga155715dd456c2b62a0b567e970af3473)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack)
[H5Eset_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga155715dd456c2b62a0b567e970af3473)
herr_t H5Eset_current_stack(hid_t err_stack_id)
Replaces the current error stack.
This function replaces the current error stack with another error stack specified by error_stack and clears the current error stack. The object handle error_stack is closed after this function call.
_To push a new error record to the error stack:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Epush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4a7d9ca6b4f7bf521d29c85bbc5b7941)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack, const char* file, const char* func,
unsigned line, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cls_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) major_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) minor_id,
const char* desc, ... )
[H5Epush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4a7d9ca6b4f7bf521d29c85bbc5b7941)
#define H5Epush
**Definition** H5version.h:1190
This function pushes a new error record onto the error stack for the current thread.
_To delete some error messages:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Epop](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga818569ac92695fb1a7836c2389faa2ba)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack, size_t count)
[H5Epop](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga818569ac92695fb1a7836c2389faa2ba)
herr_t H5Epop(hid_t err_stack, size_t count)
Deletes specified number of error messages from the error stack.
This function deletes some error messages from the error stack.
_To retrieve the number of error records:_
int [H5Eget_num](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga5c42673e2059c385a95ce3c597e0756d)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack)
[H5Eget_num](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga5c42673e2059c385a95ce3c597e0756d)
ssize_t H5Eget_num(hid_t error_stack_id)
Retrieves the number of error messages in an error stack.
This function retrieves the number of error records from an error stack.
_To clear the error stack:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Eclear_stack([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack)
This function clears the error stack.
_To close the object handle for an error stack:_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Eclose_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga41c2ed13fd6aac6e413fe7383b9090fa)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) error_stack)
[H5Eclose_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga41c2ed13fd6aac6e413fe7383b9090fa)
herr_t H5Eclose_stack(hid_t stack_id)
Closes an error stack handle.
This function closes the object handle for an error stack and releases its resources.
The example below shows how an application pushes an error record onto the default error stack.
_Example: Pushing an error message to an error stack_
// Make call to HDF5 I/O routine
if((dset_id=[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, dset_name, access_plist)) < 0)
{
// Push client error onto error stack
[H5Epush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4a7d9ca6b4f7bf521d29c85bbc5b7941)([H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455),__FILE__,FUNC,__LINE__,cls_id,
CLIENT_ERR_MAJ_IO,CLIENT_ERR_MINOR_OPEN, “H5Dopen failed”);
}
// Indicate error occurred in function
return 0;
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
The example below shows how an application registers the current error stack and creates an object handle to avoid another HDF5 function from clearing the error stack.
_Example: Registering the error stack_
if ([H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dset_id, mem_type_id, mem_space_id, file_space_id, dset_xfer_plist_id, buf) < 0)
{
// Push client error onto error stack
[H5Epush2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga17c2790837a1c1ea7e56b65d3c00a920)([H5E_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#ad7ca07d2b387a59c7e8bcab22fa57455),__FILE__,FUNC,__LINE__,cls_id,
CLIENT_ERR_MAJ_IO,CLIENT_ERR_MINOR_HDF5,
“H5Dwrite failed”);
// Preserve the error stack by assigning an object handle to it
error_stack = [H5Eget_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac66c0955a6d821a472a3a408cdc95ae6)();
// Close dataset
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dset_id);
// Replace the current error stack with the preserved one
[H5Eset_current_stack](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga155715dd456c2b62a0b567e970af3473)(error_stack);
}
return 0;
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Epush2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga17c2790837a1c1ea7e56b65d3c00a920)
herr_t H5Epush2(hid_t err_stack, const char *file, const char *func, unsigned line, hid_t cls_id, hid_t maj_id, hid_t min_id, const char *msg,...)
Pushes a new error record onto an error stack.
Previous Chapter [HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#sec_attribute) - Next Chapter [Properties and Property Lists in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#sec_plist)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


