# Python-Code-Snippets
Record special Python codes


### 1. Use `singledispatch` transforms a function into a generic function

The `singledispatch` is a function decorator, which can transform a function into a generic function, that means we can use this to implement function overloading in python, like Java.

```python
from functools import singledispatch


@singledispatch
def process(*args, **kwargs):
    pass


@process.register(str)
def broker(params):
    print("str params: {}".format(type(params)))


@process.register(list)
def broker(params):
    print("list params: {}".format(type(params)))


@process.register(dict)
def broker(params):
    print("dict params: {}".format(type(params)))


if __name__ == "__main__":
    process("foo")
    process([1, 2, 3])
    process({"foo": "bar"})
```

We can use this to make a function process any type of parameters and without `if` statement. It's mostly likes a "Strategy Mode".

### 2. Use `lru_cache` to cache function result

As it's name means, `lru_cache` can save the results of time-consuming functions to avoid repeated calculations when the same parameters are passed in, like CPU cache，but in RAM.

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def factorial(n):
    return n * factorial(n-1) if n != 1 else 1
```

### 3. Use Linux Signal to set a timeout for function execution

In UNIX API，`timer_create` can create a timer, which will send a SIGALRAM signal when expired, we can use this timer and `sigsetjmp`，`siglongjmp` to implements timeout function.

Python module has UNIX Singal implements, **but it's just only works in main thread.**

```python
import time
import signal

def handler(signum, frame):
    raise TimeoutError

def with_timeout(timeout):
    """
    This decorator is only works in main thread!
    """
    def wrapper(func):
        def set_timer(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)  # Register a function to SIGALRM
            signal.alarm(timeout)  # Set timer's expire time
            result = func(*args, **kwargs)
            signal.alarm(0)  # shutdown timer
            return result
        return set_timer
    return wrapper

@with_timeout(2)
def foo():
    time.sleep(3)
    print("biu~")

if __name__ == "__main__":
    foo()
```

### 4. Use Linux Thread to set a timeout for function execution

There has a system-call which name is `pthread_join()` in POSIX, it wait for the end of another thread, here is the function prototype:

```cpp
pthread_join(pthread_t thread, void **retval)
```

pointer `**retval` is the result of another thread, in the other world, `pthread_join` can get another thread's execution result. But unfortunately, this system-call doesn't have timeout paramters.

In Python threading, we can use `Thread().join(timeout)` to wait for the end of another thread, but we can't get it's result, and we can specify `timeout` paramters, so we can use this feature to build our timeout-function.

```python
import time
from threading import Thread


class MaxTimeExceeded(Exception):
    pass


def run_func(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def with_timeout(timeout):
    def func_wrapper(func):
        def wrapper(*args, **kwargs):

            class SavedResultThread(Thread):
                def __init__(self):
                    super(SavedResultThread, self).__init__()
                    self.result = None

                def run(self) -> None:
                    self.result = run_func(func, *args, **kwargs)

            t = SavedResultThread()
            t.start()

            t.join(timeout)

            if t.is_alive():
                # One thing, we can't stop this thread, it may be run forever
                raise MaxTimeExceeded()

            return t.result
        return wrapper
    return func_wrapper


if __name__ == "__main__":

    @with_timeout(2)
    def test_func():
        time.sleep(1)
        return "ok"

    result = test_func()
    print(result)
```

### 5. Use bit shift to replace division

In binary search algorithm, we may use `start + (last - start) // 2` to calculate mid-index, but we can use bit shift to optimiz this calculation.

```bash
# python3

In [1]: 10 >> 1
Out[1]: 5

In [2]: 9 >> 1
Out[2]: 4
```

The entire binary search implemention:

```python
def binary_search(value, sorted_list):
    start, last = 0, len(sorted_list) - 1
    while start <= last:
        mid = start + ((last - start) >> 1)
        if sorted_list[mid] == value: 
            return mid
        if sorted_list[mid] < value:
            start = mid + 1
        else:
            last = mid - 1
    return None
```

By the way, also can use bit shift to calculate multiplication, such as $ m * 2^n $

```bash
In [6]: 1 << 1   # 1 * 2
Out[6]: 2

In [7]: 5 << 2   # 5 * 2^2
Out[7]: 20
```

### 6. Create metaclass template

In some special area, such as ORM, Form-Validate, we need to create a metaclass to create our own class dynamically, so we can write a "metaclass-template" to reduce iterant labor.

```python
import inspect

def is_instance_or_subclass(val, class_):
    try:
        return issubclass(val, class_)
    except TypeError:
        return isinstance(val, class_)
        

def _get_fields(attrs, field_class):
    fields = [
        (field_name, attrs.get(field_name))
        for field_name, field_value in list(attrs.items())
        if is_instance_or_subclass(field_value, field_class)
    ]
    return fields
    

def _get_fields_by_mro(klass, field_class):
    # use __mro__ get all parents
    mro = inspect.getmro(klass)
    return sum(
        (
            _get_fields(
                getattr(base, '_declared_fields', base.__dict__),
                field_class,
            )
            for base in mro[:0:-1]
        ),
        [],
    )
    

class MetaclassTemplate(type):
    def __new__(mcs, name, parents, attrs):
        
        # ignore MetaclassTemplate's child class
        bases = [b for b in parents if isinstance(b, MetaclassTemplate)]
        if not bases:
            return super().__new__(mcs, name, parents, attrs)
    
        class_fields = _get_fields(attrs, SomeType)

        klass = super(MetaclassTemplate, mcs).__new__(mcs, name, parents, attrs)

        inherited_fields = _get_fields_by_mro(klass, SomeType)

        # register _declared_fields to class object
        klass._declared_fields = dict(class_fields + inherited_fields)

        return klass
```


## 7. Use Exponential-Backoff with retry

> Exponential backoff is an algorithm that uses feedback to multiplicatively decrease the rate of some process, in order to gradually find an acceptable rate. 

When we use `requests` Python package to send a network request, or use gRPC to execute a remote procedure call, it may returns unexpected errors, GATEWAY TIMEOUT, ABORTED etc. if this network request is very significant, we need retry.

Here is a gRPC retry with Exponential-Backoff algorithm:

```python
import time
from grpc import StatusCode, RpcError


# define retry times with different situation
MAX_RETRIES_BY_CODE = {
    StatusCode.INTERNAL: 1,
    StatusCode.ABORTED: 3,
    StatusCode.UNAVAILABLE: 5,
    StatusCode.DEADLINE_EXCEEDED: 5
}

# define MIN and MAX sleeping seconds
MIN_SLEEPING = 0.015625
MAX_SLEEPING = 1.0


class RetriesExceeded(Exception):
    """docstring for RetriesExceeded"""
    pass


def retry(f):
    def wraps(*args, **kwargs):
        retries = 0
        while True:
            try:
                return f(*args, **kwargs)
            except RpcError as e:
                # 使用e.code()获取响应码
                code = e.code()
                max_retries = MAX_RETRIES_BY_CODE.get(code)

                if max_retries is None:
                    raise e

                if retries > max_retries:
                    raise RetriesExceeded(e)

                back_off = min(MIN_SLEEPING * 2 ** retries, MAX_SLEEPING)

                retries += 1
                time.sleep(back_off)
    return wraps
```