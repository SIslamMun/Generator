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
  * [ Library header files and conditional compilation](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title0 "H1")
  * [ Global variables/structures](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title1 "H1")
  * [ Global library initialization variable](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title2 "H2")
  * [ Global serialization variable](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title3 "H2")
  * [ Global thread initialization variable](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title4 "H2")
  * [ Global key for per-thread error stacks](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title5 "H2")
  * [ Global structure and key for thread cancellation prevention](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title6 "H2")
  * [ Changes to Macro expansions](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title7 "H1")
  * [ Changes to FUNC_ENTER](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title8 "H2")
  * [ Changes to HRETURN and HRETURN_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title9 "H2")
  * [ Implementation of threadsafe functionality](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title10 "H1")
  * [ Recursive Locks](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title11 "H2")
  * [ First thread initialization](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title12 "H2")
  * [ Per-thread error stack management](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title13 "H2")
  * [ Thread Cancellation safety](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title14 "H2")
  * [ Test programs](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title15 "H1")
  * [ Data set create and write](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title16 "H2")
  * [ Test on error stack](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title17 "H2")
  * [ Test on cancellation safety](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title18 "H2")
  * [ Test on attribute creation](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title19 "H2")
  * [ Appendix](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title20 "H1")
  * [ Recursive Lock implementation code](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title21 "H2")
  * [ First thread initialization](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title22 "H2")
  * [ Per-thread error stack acquisition](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title23 "H2")
  * [ Thread cancellation mechanisms](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title24 "H2")
  * [ Macro expansion codes](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#title25 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Thread Safe Library
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Library header files and conditional compilation
The following code is placed at the beginning of H5private.h: 
#ifdef H5_HAVE_THREADSAFE
#include <pthread.h>
#endif
`H5_HAVE_THREADSAFE` is defined when the HDF5 library is compiled with the _HDF5_ENABLE_THREADSAFE=ON_ using CMake. In general, code for the non-threadsafe version of HDF5 library are placed within the `# else` part of the conditional compilation. The exception to this rule are the changes to the `FUNC_ENTER` (in H5private.h), `HRETURN` and `HRETURN_ERROR` (in H5Eprivate.h) macros (see section [Changes to HRETURN and HRETURN_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_macro_ret)). 
#  Global variables/structures
##  Global library initialization variable
In the threadsafe implementation, the global library initialization variable `H5_libinit_g` is changed to a global structure consisting of the variable with its associated lock (locks are explained in section [Recursive Locks](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_locks)): 
bool H5_libinit_g = FALSE;
becomes 
H5_api_t H5_g;
where `H5_api_t` is 
typedef struct H5_api_struct {
H5_mutex_t init_lock; // API entrance mutex
bool H5_libinit_g;
} H5_api_t;
All former references to `H5_libinit_g` in the library are now made using the macro `H5_INIT_GLOBAL`. If the threadsafe library is to be used, the macro is set to `H5_g.H5_libinit_g` instead.
##  Global serialization variable
A new global boolean variable `H5_allow_concurrent_g` is used to determine if multiple threads are allowed to an API call simultaneously. This is set to `FALSE`.
All APIs that are allowed to do so have their own local variable that shadows the global variable and is set to `TRUE`. In phase 1, no such APIs exist.
It is defined in `H5.c` as follows: 
bool H5_allow_concurrent_g = FALSE;
##  Global thread initialization variable
The global variable `H5_first_init_g` of type `pthread_once_t` is used to allow only the first thread in the application process to call an initialization function using `pthread_once`. All subsequent calls to `pthread_once` by any thread are disregarded.
The call sets up the mutex in the global structure `H5_g` (see section [Global library initialization variable](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_global_var)) via an initialization function `H5_first_thread_init`. The first thread initialization function is described in section [First thread initialization](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_first).
`H5_first_init_g` is defined in `H5.c` as follows: 
pthread_once_t H5_first_init_g = PTHREAD_ONCE_INIT;
##  Global key for per-thread error stacks
A global pthread-managed key `H5_errstk_key_g` is used to allow pthreads to maintain a separate error stack (of type `H5E_t`) for each thread. This is defined in `H5.c` as: 
pthread_key_t H5_errstk_key_g;
Error stack management is described in section [Per-thread error stack management](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_err).
##  Global structure and key for thread cancellation prevention
We need to preserve the thread cancellation status of each thread individually by using a key `H5_cancel_key_g`. The status is preserved using a structure (of type `H5_cancel_t`) which maintains the cancellability state of the thread before it entered the library and a count (which works very much like the recursive lock counter) which keeps track of the number of API calls the thread makes within the library.
The structure is defined in `H5private.h` as: 
// cancellability structure
typedef struct H5_cancel_struct {
int previous_state;
unsigned int cancel_count;
} H5_cancel_t;
Thread cancellation is described in section [Thread Cancellation safety](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_cancel).
#  Changes to Macro expansions
##  Changes to FUNC_ENTER
The `FUNC_ENTER` macro is now extended to include macro calls to initialize first threads, disable cancellability and wraps a lock operation around the checking of the global initialization flag. It should be noted that the cancellability should be disabled before acquiring the lock on the library. Doing so otherwise would allow the possibility that the thread be cancelled just after it has acquired the lock on the library and in that scenario, if the cleanup routines are not properly set, the library would be permanently locked out.
The additional macro code and new macro definitions can be found in Appendix [Macro expansion codes](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_app_E). The changes are made in `H5private.h`.
##  Changes to HRETURN and HRETURN_ERROR
The `HRETURN` and `HRETURN_ERROR` macros are the counterparts to the `FUNC_ENTER` macro described in section [Changes to FUNC_ENTER](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_macro_fe). `FUNC_LEAVE` makes a macro call to `HRETURN`, so it is also covered here.
The basic changes to these two macros involve adding macro calls to call an unlock operation and re-enable cancellability if necessary. It should be noted that the cancellability should be re-enabled only after the thread has released the lock to the library. The consequence of doing otherwise would be similar to that described in section [Changes to FUNC_ENTER](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_macro_fe).
The additional macro code and new macro definitions can be found in Appendix [Macro expansion codes](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_app_E). The changes are made in `H5Eprivate.h`.
#  Implementation of threadsafe functionality
##  Recursive Locks
A recursive mutex lock m allows a thread t1 to successfully lock m more than once without blocking t1. Another thread t2 will block if t2 tries to lock m while t1 holds the lock to m. If t1 makes k lock calls on m, then it also needs to make k unlock calls on m before it releases the lock.
Our implementation of recursive locks is built on top of a pthread mutex lock (which is not recursive). It makes use of a pthread condition variable to have unsuccessful threads wait on the mutex. Waiting threads are awaken by a signal from the final unlock call made by the thread holding the lock.
Recursive locks are defined to be the following type (`H5private.h`): 
typedef struct H5_mutex_struct {
pthread_t owner_thread; // current lock owner
pthread_mutex_t atomic_lock; // lock for atomicity of new mechanism
pthread_cond_t cond_var; // condition variable
unsigned int lock_count;
} H5_mutex_t;
Detailed implementation code can be found in Appendix [Recursive Lock implementation code](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_app_A). The implementation changes are made in `H5TS.c`.
##  First thread initialization
Because the mutex lock associated with a recursive lock cannot be statically initialized, a mechanism is required to initialize the recursive lock associated with `H5_g` so that it can be used for the first time.
The pthreads library allows this through the pthread_once call which as described in section [Global thread initialization variable](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_global_init) allows only the first thread accessing the library in an application to initialize `H5_g`.
In addition to initializing `H5_g`, it also initializes the key (see section [Global key for per-thread error stacks](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_global_key)) for use with per-thread error stacks (see section [Per-thread error stack management](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_err)).
The first thread initialization mechanism is implemented as the function call `H5_first_thread_init()` in `H5TS.c`. This is described in appendix B.
##  Per-thread error stack management
Pthreads allows individual threads to access dynamic and persistent per-thread data through the use of keys. Each key is associated with a table that maps threads to data items. Keys can be initialized by `pthread_key_create()` in pthreads (see sections [Global key for per-thread error stacks](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_global_key) and [First thread initialization](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_first)). Per-thread data items are accessed using a key through the `pthread_getspecific()` and `pthread_setspecific()` calls to read and write to the association table respectively.
Per-thread error stacks are accessed through the key `H5_errstk_key_g` which is initialized by the first thread initialization call (see section [First thread initialization](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_impl_first)).
In the non-threadsafe version of the library, there is a global stack variable `H5E_stack_g[1]` which is no longer defined in the threadsafe version. At the same time, the macro call to gain access to the error stack `H5E_get_my_stack` is changed from: 
#define H5E_get_my_stack() (H5E_stack_g+0)
to: 
#define H5E_get_my_stack() H5E_get_stack()
where `H5E_get_stack()` is a surrogate function that does the following operations: 
  1. if a thread is attempting to get an error stack for the first time, the error stack is dynamically allocated for the thread and associated with `H5_errstk_key_g` using `pthread_setspecific()`. The way we detect if it is the first time is through `pthread_getspecific()` which returns `NULL` if no previous value is associated with the thread using the key.
  2. if `pthread_getspecific()` returns a non-null value, then that is the pointer to the error stack associated with the thread and the stack can be used as usual. 


A final change to the error reporting routines is as follows; the current implementation reports errors to always be detected at thread 0. In the threadsafe implementation, this is changed to report the number returned by a call to `pthread_self()`.
The change in code (reflected in `H5Eprint` of file `H5E.c`) is as follows: 
#ifdef H5_HAVE_THREADSAFE
fprintf (stream, "HDF5-DIAG: Error detected in thread %d." ,pthread_self());
#else
fprintf (stream, "HDF5-DIAG: Error detected in thread 0.");
#endif
Code for `H5E_get_stack()` can be found in Appendix [Per-thread error stack acquisition](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_app_C). All the above changes were made in `H5E.c`.
##  Thread Cancellation safety
To prevent thread cancellations from killing a thread while it is in the library, we maintain per-thread information about the cancellability status of the thread before it entered the library so that we can restore that same status when the thread leaves the library.
By _enter_ and _leave_ the library, we mean the points when a thread makes an API call from a user application and the time that API call returns. Other API or callback function calls made from within that API call are considered _within_ the library.
Because other API calls may be made from within the first API call, we need to maintain a counter to determine which was the first and correspondingly the last return.
When a thread makes an API call, the macro `H5_API_SET_CANCEL` calls the worker function `H5_cancel_count_inc()` which does the following: 
  1. if this is the first time the thread has entered the library, a new cancellability structure needs to be assigned to it. 
  2. if the thread is already within the library when the API call is made, then cancel_count is simply incremented. Otherwise, we set the cancellability state to `PTHREAD_CANCEL_DISABLE` while storing the previous state into the cancellability structure. `cancel_count` is also incremented in this case. 


When a thread leaves an API call, the macro `H5_API_UNSET_CANCEL` calls the worker function `H5_cancel_count_dec()` which does the following: 
  1. if `cancel_count` is greater than 1, indicating that the thread is not yet about to leave the library, then `cancel_count` is simply decremented. 
  2. otherwise, we reset the cancellability state back to its original state before it entered the library and decrement the count (back to zero). 


`H5_cancel_count_inc` and `H5_cancel_count_dec` are described in Appendix [Thread cancellation mechanisms](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html#subsec_tsafe_app_D) and may be found in `H5TS.c`.
#  Test programs
Except where stated, all tests involve 16 simultaneous threads that make use of HDF5 API calls without any explicit synchronization typically required in a non-threadsafe environment.
##  Data set create and write
The test program sets up 16 threads to simultaneously create 16 different datasets named from _zero_ to _fifteen_ for a single file and then writing an integer value into that dataset equal to the dataset's named value.
The main thread would join with all 16 threads and attempt to match the resulting HDF5 file with expected results - that each dataset contains the correct value (0 for _zero_ , 1 for _one_ etc ...) and all datasets were correctly created.
The test is implemented in the file `ttsafe_dcreate.c`.
##  Test on error stack
The error stack test is one in which 16 threads simultaneously try to create datasets with the same name. The result, when properly serialized, should be equivalent to 16 attempts to create the dataset with the same name.
The error stack implementation runs correctly if it reports 15 instances of the dataset name conflict error and finally generates a correct HDF5 containing that single dataset. Each thread should report its own stack of errors with a thread number associated with it.
The test is implemented in the file `ttsafe_error.c`.
##  Test on cancellation safety
The main idea in thread cancellation safety is as follows; a child thread is spawned to create and write to a dataset. Following that, it makes a `H5Diterate` call on that dataset which activates a callback function.
A deliberate barrier is invoked at the callback function which waits for both the main and child thread to arrive at that point. After that happens, the main thread proceeds to make a thread cancel call on the child thread while the latter sleeps for 3 seconds before proceeding to write a new value to the dataset.
After the iterate call, the child thread logically proceeds to wait another 3 seconds before writing another newer value to the dataset.
The test is correct if the main thread manages to read the second value at the end of the test. This means that cancellation did not take place until the end of the iteration call despite of the 3 second wait within the iteration callback and the extra dataset write operation. Furthermore, the cancellation should occur before the child can proceed to write the last value into the dataset.
##  Test on attribute creation
A main thread makes 16 threaded calls to `H5Acreate` with a generated name for each attribute. Sixteen attributes should be created for the single dataset in random (chronological) order and receive values depending on its generated attribute name (e.g. _attrib010_ would receive the value 10).
After joining with all child threads, the main thread proceeds to read each attribute by generated name to see if the value tallies. Failure is detected if the attribute name does not exist (meaning they were never created) or if the wrong values were read back.
#  Appendix
##  Recursive Lock implementation code
void H5_mutex_init(H5_mutex_t *H5_mutex)
{
H5_mutex-&gt;owner_thread = NULL;
pthread_mutex_init(&amp;H5_mutex-&gt;atomic_lock, NULL);
pthread_cond_init(&amp;H5_mutex-&gt;cond_var, NULL);
H5_mutex-&gt;lock_count = 0;
}
void H5_mutex_lock(H5_mutex_t *H5_mutex)
{
pthread_mutex_lock(&amp;H5_mutex-&gt;atomic_lock);
if (pthread_equal(pthread_self(), H5_mutex-&gt;owner_thread)) {
// already owned by self - increment count
H5_mutex-&gt;lock_count++;
} 
else {
if (H5_mutex-&gt;owner_thread == NULL) {
// no one else has locked it - set owner and grab lock
H5_mutex-&gt;owner_thread = pthread_self();
H5_mutex-&gt;lock_count = 1;
} 
else {
/* if already locked by someone else */
while (1) {
pthread_cond_wait(&amp;H5_mutex-&gt;cond_var, &amp;H5_mutex-&gt;atomic_lock);
if (H5_mutex-&gt;owner_thread == NULL) {
H5_mutex-&gt;owner_thread = pthread_self();
H5_mutex-&gt;lock_count = 1;
break;
} // else do nothing and loop back to wait on condition
}
}
}
pthread_mutex_unlock(&amp;H5_mutex-&gt;atomic_lock);
}
void H5_mutex_unlock(H5_mutex_t *H5_mutex)
{
pthread_mutex_lock(&amp;H5_mutex-&gt;atomic_lock);
H5_mutex-&gt;lock_count--;
if (H5_mutex-&gt;lock_count == 0) {
H5_mutex-&gt;owner_thread = NULL;
pthread_cond_signal(&amp;H5_mutex-&gt;cond_var);
}
pthread_mutex_unlock(&amp;H5_mutex-&gt;atomic_lock);
}
##  First thread initialization
void H5_first_thread_init(void)
{
// initialize global API mutex lock
H5_g.H5_libinit_g = FALSE;
H5_g.init_lock.owner_thread = NULL;
pthread_mutex_init(&amp;H5_g.init_lock.atomic_lock, NULL);
pthread_cond_init(&amp;H5_g.init_lock.cond_var, NULL);
H5_g.init_lock.lock_count = 0;
// initialize key for thread-specific error stacks
pthread_key_create(&amp;H5_errstk_key_g, NULL);
// initialize key for thread cancellability mechanism
pthread_key_create(&amp;H5_cancel_key_g, NULL);
}
##  Per-thread error stack acquisition
H5E_t *H5E_get_stack(void)
{
H5E_t *estack;
if (estack = pthread_getspecific(H5_errstk_key_g)) {
return estack;
} 
else {
// no associated value with current thread - create one
estack = (H5E_t *)malloc(sizeof(H5E_t));
pthread_setspecific(H5_errstk_key_g, (void *)estack);
return estack;
}
}
##  Thread cancellation mechanisms
void H5_cancel_count_inc(void)
{
H5_cancel_t *cancel_counter;
if (cancel_counter = pthread_getspecific(H5_cancel_key_g)) {
// do nothing here
} 
else {
// first time thread calls library - create new counter and
// associate with key
cancel_counter = (H5_cancel_t *)malloc(sizeof(H5_cancel_t));
cancel_counter-&gt;cancel_count = 0;
pthread_setspecific(H5_cancel_key_g, (void *)cancel_counter);
}
if (cancel_counter-&gt;cancel_count == 0) {
/* thread entering library */
pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &amp;(cancel_counter-&gt;previous_state));
}
cancel_counter-&gt;cancel_count++;
}
void H5_cancel_count_dec(void)
{
H5_cancel_t *cancel_counter = pthread_getspecific(H5_cancel_key_g);
if (cancel_counter-&gt;cancel_count == 1)
pthread_setcancelstate(cancel_counter-&gt;previous_state, NULL);
cancel_counter-&gt;cancel_count--;
}
##  Macro expansion codes
###  `FUNC_ENTER`
// Initialize the library \
H5_FIRST_THREAD_INIT \
H5_API_UNSET_CANCEL \
H5_API_LOCK_BEGIN \
if (!(H5_INIT_GLOBAL)) { \
H5_INIT_GLOBAL = TRUE; \
if (H5_init_library() &lt; 0) { \
HRETURN_ERROR (H5E_FUNC, H5E_CANTINIT, err, \
"library initialization failed"); \
} \
} \
H5_API_LOCK_END \
:
:
:
###  `H5_FIRST_THREAD_INIT`
// Macro for first thread initialization
#define H5_FIRST_THREAD_INIT \
pthread_once(&amp;H5_first_init_g, H5_first_thread_init);
###  `H5_API_UNSET_CANCEL`
#define H5_API_UNSET_CANCEL \
if (H5_IS_API(__func__)) { \
H5_cancel_count_inc(); \
}
###  `H5_API_LOCK_BEGIN`
#define H5_API_LOCK_BEGIN \
if (H5_IS_API(__func__)) { \
H5_mutex_lock(&amp;H5_g.init_lock);
###  `H5_API_LOCK_END`
#define H5_API_LOCK_END }
###  `HRETURN` and `HRETURN_ERROR`
:
:
H5_API_UNLOCK_BEGIN \
H5_API_UNLOCK_END \
H5_API_SET_CANCEL \
return ret_val; \
}
###  `H5_API_UNLOCK_BEGIN`
#define H5_API_UNLOCK_BEGIN \
if (H5_IS_API(__func__)) { \
H5_mutex_unlock(&amp;H5_g.init_lock);
###  `H5_API_UNLOCK_END`
#define H5_API_UNLOCK_END }
</pre>
###  `H5_API_SET_CANCEL`
#define H5_API_SET_CANCEL \
if (H5_IS_API(__func__)) { \
H5_cancel_count_dec(); \
}
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


