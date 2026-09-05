# Agentic AI Learning — Notes

Everything covered so far, in the order it was learned: **plain Python → advanced Python →
data libraries → Pydantic → LangChain → RAG**. Every section links to the notebook it came
from, so the notes and the runnable code stay in sync.

---

## Table of contents

**Part 1 — Python**
1. [Syntax, variables and data types](#1-syntax-variables-and-data-types)
2. [Control flow](#2-control-flow)
3. [Data structures](#3-data-structures)
4. [Functions](#4-functions)
5. [Object-oriented programming](#5-object-oriented-programming)
6. [Magic methods and operator overloading](#6-magic-methods-and-operator-overloading)
7. [Iterators and generators](#7-iterators-and-generators)
8. [Decorators](#8-decorators)
9. [Exception handling](#9-exception-handling)
10. [Modules, packages and the standard library](#10-modules-packages-and-the-standard-library)
11. [File handling and paths](#11-file-handling-and-paths)
12. [Logging](#12-logging)
13. [NumPy](#13-numpy)
14. [Pandas](#14-pandas)

**Part 2 — GenAI / LangChain**
15. [Pydantic](#15-pydantic)
16. [LangChain setup and LCEL chains](#16-langchain-setup-and-lcel-chains)
17. [Document loaders](#17-document-loaders)
18. [Text splitters](#18-text-splitters)
19. [Embeddings](#19-embeddings)
20. [Vector stores](#20-vector-stores)
21. [Building a RAG pipeline](#21-building-a-rag-pipeline)
22. [Serving it with Streamlit](#22-serving-it-with-streamlit)
23. [Gotchas worth remembering](#23-gotchas-worth-remembering)

---

# Part 1 — Python

## 1. Syntax, variables and data types

> Notebooks: [01-python-basics](01-python/01-basics/01-python-basics.ipynb) ·
> [02-variables](01-python/01-basics/02-variables.ipynb) ·
> [03-data-types](01-python/01-basics/03-data-types.ipynb)

Python is **case sensitive** and uses **indentation** instead of braces to mark blocks. All
statements in a block must sit at the same indent level.

```python
age = 31

if age >= 30:
    print("Age is older than 30")
else:
    print("I can't print anything")
```

A long expression can be continued on the next line with a backslash:

```python
total = 1 + 2 + 3 + 4 + 5 \
        + 6 + 7
print(total)   # 28
```

Python is **dynamically typed** — the interpreter works out the type at runtime, and the same
name can be rebound to a different type:

```python
var = 10
print(type(var))    # <class 'int'>

var = "Shaurya"
print(type(var))    # <class 'str'>
```

### Variables

Variables are created by assignment; there is no separate declaration step.

```python
name   = "Shaurya"    # str
age    = 21           # int
cgpa   = 8.53         # float
single = True         # bool
```

Naming rules: start with a letter or underscore, then letters/digits/underscores. Names are
case sensitive. Convention is `snake_case` and descriptive (`first_name`, not `fn`).

### Type conversion and input

```python
age = 21
age_str = str(age)
print(type(age_str))   # <class 'str'>

# input() always returns a string - cast it yourself
age = int(input("Enter your age: "))
```

The four core scalar types are `int`, `float`, `str`, `bool`. Types matter because they decide
which operations are legal, what range of values fits, and how much memory is used.

---

## 2. Control flow

> Notebooks: [04-conditional-statements](01-python/01-basics/04-conditional-statements.ipynb) ·
> [05-loops](01-python/01-basics/05-loops.ipynb)

`if` / `elif` / `else`, and they nest:

```python
num = int(input("Enter a number: "))

if num > 0:
    print("The number is positive.")
    if num % 2 == 0:
        print("The number is even.")
    else:
        print("The number is odd.")
elif num < 0:
    print("The number is negative.")
else:
    print("The number is zero.")
```

`for` walks an iterable, `while` repeats until a condition fails, `break` exits early:

```python
for i in range(1, 11):
    print(i)

i = 1
while i < 11:
    print(i)
    i += 1

num = 1
while True:
    print(num)
    num += 1
    if num > 10:
        break
```

`range(start, stop, step)` — `stop` is **exclusive**.

---

## 3. Data structures

### Lists — ordered, mutable

> Notebook: [06-lists](01-python/01-basics/06-lists.ipynb)

```python
lst = [1, 2, 3, 4, 5, 6, 7]

for i in range(len(lst)):
    print(lst[i])

# building a list from input
lst = []
for _ in range(10):
    lst.append(int(input("Enter a number: ")))
```

### Tuples — ordered, **immutable**

> Notebook: [07-tuples](01-python/01-basics/07-tuples.ipynb)

```python
tup = (1, 2, 3, 4, 5)
print(tup[0])    # 1   - positive index
print(tup[-1])   # 5   - negative index counts from the end
```

A one-element tuple needs a trailing comma, otherwise the parentheses are just grouping:

```python
t = (5)     # int!
t = (5,)    # tuple
```

Only two methods, because tuples cannot change:

```python
t = (1, 2, 2, 3, 2)
print(t.count(2))   # 3
print(t.index(3))   # 3
```

Packing / unpacking — this is what makes the one-line swap work:

```python
student = "Rahul", 21, "Delhi"        # packing
name, age, city = student             # unpacking

a, b = 10, 20
a, b = b, a                           # swap, no temp variable
```

Concatenation, repetition, membership, and sorting (which returns a **list**):

```python
(1, 2) + (3, 4)     # (1, 2, 3, 4)
(1, 2) * 3          # (1, 2, 1, 2, 1, 2)
20 in (10, 20, 30)  # True
sorted((10, 20, 5)) # [5, 10, 20]  -> a list, not a tuple

list((1, 2, 3))     # tuple -> list
tuple([1, 2, 3])    # list  -> tuple
```

Tuples nest, and indexing chains:

```python
student = ("Rahul", (90, 85, 88))
print(student[1][0])    # 90
```

### Sets — unordered, unique

> Notebook: [08-sets](01-python/01-basics/08-sets.ipynb)

Duplicates are dropped automatically, which is the whole point:

```python
numbers = {1, 2, 3, 3, 4, 4, 5, 5}
print(numbers)                        # {1, 2, 3, 4, 5}

names = ["Rahul", "Aman", "Rahul", "Priya", "Aman"]
print(set(names))                     # {'Rahul', 'Aman', 'Priya'}
```

`{}` creates an empty **dict**, not a set — use `set()`:

```python
s = set()          # empty set
d = {}             # empty dict
```

Mutating a set:

```python
s = {1, 2, 3}
s.add(4)             # add one
s.update([5, 6])     # add many
s.remove(2)          # KeyError if missing
x = s.pop()          # removes an arbitrary element (sets are unordered)
print(3 in s)        # membership test - O(1)
```

Set algebra:

```python
A, B = {1, 2, 3, 4}, {3, 4, 5, 6}

A | B    # union                {1,2,3,4,5,6}   (A.union(B))
A & B    # intersection         {3,4}           (A.intersection(B))
A - B    # difference           {1,2}
A ^ B    # symmetric difference {1,2,5,6}

{1, 2}.issubset(B)        # True
B.issuperset({3, 4})      # True
{1, 2}.isdisjoint({3, 4}) # True - no common elements
```

The classic use: deduplicate a list.

```python
nums = [1, 2, 2, 3, 4, 4]
unique = list(set(nums))   # [1, 2, 3, 4]
```

### Dictionaries — key/value pairs

> Notebook: [09-dictionaries](01-python/01-basics/09-dictionaries.ipynb)

```python
student = {"name": "Shaurya", "age": 21, "isPlaced": False}
student = dict(name="Shaurya", age=21, isPlaced=False)   # same thing
```

`[]` raises `KeyError` on a missing key; `.get()` returns `None` (or a default) instead:

```python
student["name"]        # 'Shaurya'
student.get("cgpa")    # None      - no exception
student.get("cgpa", 0) # 0         - with a default
```

Dicts nest, which is how JSON-shaped data is modelled:

```python
person = {
    "name": "Shaurya",
    "address": {"street": "123 Main St", "city": "New York"},
}
print(person["address"]["city"])   # New York
```

Iterating and the common methods:

```python
for key in student:                     # iterates keys
    print(key)

for key, value in student.items():      # the usual way
    print(key, value)

student.keys()      # view of keys
student.values()    # view of values
"name" in student   # membership checks the KEYS
student.pop("age")  # remove by key
student.popitem()   # remove and return the last inserted pair
```

Dict comprehension:

```python
square = {x: x * x for x in range(5) if x % 2 == 0}   # {0: 0, 2: 4, 4: 16}
```

Merging (Python 3.9+):

```python
{"a": 1} | {"b": 2}    # {'a': 1, 'b': 2}
```

Two patterns worth memorising — **frequency count** and **invert a dict**:

```python
arr = [1, 2, 2, 3, 3, 3]

freq = {}
for x in arr:
    freq[x] = freq.get(x, 0) + 1        # the short way
print(freq)                             # {1: 1, 2: 2, 3: 3}

d = {"a": 1, "b": 2}
reverse = {v: k for k, v in d.items()}  # {1: 'a', 2: 'b'}
```

### Quick comparison

| Type   | Ordered | Mutable | Duplicates | Syntax |
|--------|---------|---------|------------|--------|
| list   | yes     | yes     | yes        | `[1, 2]` |
| tuple  | yes     | no      | yes        | `(1, 2)` |
| set    | no      | yes     | no         | `{1, 2}` |
| dict   | yes (insertion order) | yes | keys unique | `{"a": 1}` |

---

## 4. Functions

> Notebook: [10-functions](01-python/01-basics/10-functions.ipynb)

A function is a named, reusable block of code.

```python
def area(length, breadth):
    return length * breadth

print(area(10, 20))    # 200
```

### Default and keyword arguments

```python
def greet(name="Guest"):
    print("Hello", name)

greet()             # Hello Guest
greet("Shaurya")    # Hello Shaurya

def student(name, age):
    print(name, age)

student(age=20, name="Shaurya")   # order doesn't matter with keywords
```

### `*args` and `**kwargs`

`*args` collects extra positional arguments into a **tuple**; `**kwargs` collects extra keyword
arguments into a **dict**.

```python
def total(*numbers):
    print(sum(numbers))

total(1, 2, 3, 4, 8)     # 18

def student(**info):
    print(info)

student(name="Shaurya", age=20)   # {'name': 'Shaurya', 'age': 20}
```

### Lambda — anonymous one-expression functions

```python
square   = lambda x: x ** 2
multiply = lambda a, b: a * b

print(square(5))       # 25
print(multiply(5, 6))  # 30
```

Use one when a function is needed *once* and naming it would add nothing. Instead of writing
`def cube(x): return x ** 3`, just write `lambda x: x ** 3` inline.

### `map` and `filter`

`map(fn, iterable)` applies `fn` to every element. `filter(fn, iterable)` keeps elements where
`fn` returns `True`. Both are **lazy** — wrap in `list()` to see the result.

```python
numbers = [1, 2, 3, 4]
list(map(lambda x: x * x, numbers))          # [1, 4, 9, 16]
list(map(str.upper, ["rahul", "aman"]))      # ['RAHUL', 'AMAN']

numbers = [1, 2, 3, 4, 5, 6]
list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6]
list(filter(lambda x: x > 10, [5, 12, 18]))  # [12, 18]

# chained: square only the even numbers
even   = filter(lambda x: x % 2 == 0, numbers)
result = map(lambda x: x * x, even)
print(list(result))                          # [4, 16, 36]
```

A named function works just as well as a lambda — `map(square, numbers)`.

---

## 5. Object-oriented programming

> Notebook: [01-classes-and-oop](01-python/02-oop/01-classes-and-oop.ipynb)

A **class** is a blueprint; an **object** is an instance built from it.

```python
class Car:
    pass

audi = Car()
bmw  = Car()
print(type(audi))    # <class '__main__.Car'>
```

### Constructor, instance variables, instance methods

`__init__` runs on creation. `self` is the instance the method was called on.

```python
class Dog:
    def __init__(self, name, age):   # constructor
        self.name = name             # instance variables
        self.age = age

    def bark(self):                  # instance method
        print(f"{self.name} says woof woof")

dog1 = Dog("Tommy", 2)
dog1.bark()      # Tommy says woof woof
```

A fuller example — state plus the methods that guard it:

```python
class BankAccount:
    def __init__(self, account_number, account_holder, balance):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{amount} deposited. New balance is {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
        else:
            self.balance -= amount
            print(f"{amount} withdrawn. New balance is {self.balance}")

    def get_balance(self):
        print(f"Current balance is {self.balance}")

account = BankAccount("123456", "shaurya", 1000)
account.deposit(500)
account.withdraw(200)
account.get_balance()
```

### Inheritance

A child class inherits attributes and methods from its parent. `super().__init__(...)` runs the
parent's constructor so you don't repeat the assignments.

```python
class Car:
    def __init__(self, windows, doors, enginetype):
        self.windows = windows
        self.doors = doors
        self.enginetype = enginetype

    def drive(self):
        print("drives with engine type", self.enginetype)

class Tesla(Car):
    def __init__(self, windows, doors, enginetype, autopilot):
        super().__init__(windows, doors, enginetype)   # reuse the parent
        self.autopilot = autopilot

    def drive(self):                                    # override
        print("drives with engine type", self.enginetype,
              "and autopilot is", self.autopilot)

Tesla(4, 4, "electric", True).drive()
```

**Multiple inheritance** — more than one base class. Each parent constructor is called explicitly:

```python
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        print(f"{self.name} makes a sound")

class Pet:
    def __init__(self, owner):
        self.owner = owner

class Dog(Animal, Pet):
    def __init__(self, name, owner):
        Animal.__init__(self, name)
        Pet.__init__(self, owner)

    def speak(self):
        print(f"{self.name} says woof and is owned by {self.owner}")
```

### Polymorphism

The same call works on different types because each one supplies its own implementation.
Achieved through **method overriding**:

```python
class Animal:
    def speak(self):
        print("Animal makes a sound")

class Dog(Animal):
    def speak(self):
        print("Dog says woof woof")

class Cat(Animal):
    def speak(self):
        print("Cat says meow meow")
```

The payoff is code that works against the *base* type:

```python
class Shape:
    def area(self):
        print("The area of the shape is calculated")

class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side * self.side

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14 * self.radius * self.radius

def print_area(shape):        # doesn't care which subclass it gets
    print(shape.area())

print_area(Square(5))     # 25
print_area(Circle(3))     # 28.26
```

### Abstract base classes

An ABC defines the interface and **forces** subclasses to implement it — instantiating a
subclass that skips an `@abstractmethod` raises `TypeError`.

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def startengine(self):
        pass

class Car(Vehicle):
    def startengine(self):
        print("Car engine started")

class Motorcycle(Vehicle):
    def startengine(self):
        print("Motorcycle engine started")

Car().startengine()
```

### Encapsulation — private and protected attributes

`__name` (two underscores) is **private**: Python name-mangles it to `_Person__name`, so it is
not reachable as `person.__name`. Access goes through getters/setters, which lets the class
validate changes.

```python
class Person:
    def __init__(self, name, age):
        self.__name = name          # private
        self.__age = age

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    def set_age(self, age):
        if age > 0:                  # the class stays in control of its state
            self.__age = age
        else:
            print("Age must be positive")

person = Person("John", 30)
person.set_age(31)
print(person.get_age())    # 31
```

`_name` (one underscore) is **protected** — a *convention* only. It signals "internal, don't
touch from outside", but Python will not stop you, and subclasses read it freely via `self._name`.

| Form     | Meaning   | Enforced? |
|----------|-----------|-----------|
| `name`   | public    | —         |
| `_name`  | protected | no, convention only |
| `__name` | private   | yes, via name mangling |

**Abstraction** is the other half: expose only what a caller needs (`deposit`, `withdraw`) and
hide how it works inside.

---

## 6. Magic methods and operator overloading

> Notebooks: [02-magic-methods](01-python/02-oop/02-magic-methods.ipynb) ·
> [03-operator-overloading](01-python/02-oop/03-operator-overloading.ipynb)

Magic (dunder) methods start and end with `__`. Overriding them lets your objects respond to
built-in syntax — `len()`, `[]`, `print()`, `+`, `==`, and so on.

| Method        | Triggered by             |
|---------------|--------------------------|
| `__init__`    | `Person(...)`            |
| `__str__`     | `print(obj)`, `str(obj)` |
| `__repr__`    | the REPL, debugging      |
| `__len__`     | `len(obj)`               |
| `__getitem__` | `obj[i]`                 |
| `__setitem__` | `obj[i] = x`             |

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age!r})"

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        return self.name[index]

    def __setitem__(self, index, value):
        chars = list(self.name)
        chars[index] = value
        self.name = "".join(chars)

person = Person("Alice", 30)
print(person[0])    # A
print(len(person))  # 5
print(person)       # Person(name=Alice, age=30)
```

`dir(Person)` lists every method available, dunders included — a quick way to see what a class
already supports.

### Operator overloading

| Method         | Operator |
|----------------|----------|
| `__add__`      | `+`      |
| `__sub__`      | `-`      |
| `__mul__`      | `*`      |
| `__truediv__`  | `/`      |
| `__eq__`       | `==`     |
| `__gt__`       | `>`      |
| `__iter__`     | `for ... in obj` |

A vector class that behaves like a number:

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    __str__ = __repr__            # reuse repr for print()

    def __add__(self, other):      return Vector(self.x + other.x, self.y + other.y)
    def __sub__(self, other):      return Vector(self.x - other.x, self.y - other.y)
    def __mul__(self, other):      return Vector(self.x * other.x, self.y * other.y)
    def __truediv__(self, other):  return Vector(self.x / other.x, self.y / other.y)
    def __eq__(self, other):       return self.x == other.x and self.y == other.y
    def __gt__(self, other):       return self.x > other.x and self.y > other.y

    def __iter__(self):
        yield self.x
        yield self.y

v1, v2 = Vector(2, 3), Vector(4, 5)
print(v1 + v2)     # Vector(6, 8)
print(v1 - v2)     # Vector(-2, -2)
print(v1 * v2)     # Vector(8, 15)
print(v1 / v2)     # Vector(0.5, 0.6)
print(v1 == v2)    # False
print(v1 > v2)     # False
```

---

## 7. Iterators and generators

> Notebooks: [01-iterators](01-python/03-advanced-python/01-iterators.ipynb) ·
> [02-generators](01-python/03-advanced-python/02-generators.ipynb)

### Iterators

An iterator hands back elements one at a time without exposing the underlying structure. This
is **lazy loading** — nothing is produced until it is asked for.

```python
my_list = [1, 2, 3, 4, 5, 6]

iterator = iter(my_list)     # get an iterator from an iterable
print(next(iterator))        # 1
print(next(iterator))        # 2

try:
    print(next(iterator))
except StopIteration:
    print("No more elements in the iterator.")
```

A `for` loop is just `iter()` + repeated `next()` + catching `StopIteration`.

### Generators

A generator is the easy way to write an iterator: use `yield` instead of `return`.

```python
def squares(n):
    for i in range(n):
        yield i ** 2

for value in squares(3):
    print(value)      # 0, 1, 4
```

- Each `yield` **pauses** the function and remembers exactly where it stopped.
- The next request resumes from that point, with all local state intact.
- Values are produced on demand, so nothing large is held in memory.

**Why it matters:** less memory, faster startup, and you can model infinite sequences. It's the
same idea LangChain uses when it streams tokens instead of buffering a whole response.

---

## 8. Decorators

> Notebook: [03-decorators](01-python/03-advanced-python/03-decorators.ipynb)

A decorator adds behaviour to a function *without editing the function's own code*. Three ideas
build up to it.

**1. Functions are objects** — you can assign them to another name:

```python
def welcome():
    return "Welcome to Python Programming"

wel = welcome     # no parentheses - the function itself, not its result
wel()             # 'Welcome to Python Programming'
```

**2. Closures** — a function defined inside another function, which captures the outer variables:

```python
def main_welcome(msg):
    def sub_welcome():
        print(msg)                     # `msg` is captured from the enclosing scope
        print("Welcome to inner function")
    return sub_welcome                 # return the function, don't call it

main_welcome("Hello Python Learners")()
```

**3. A decorator** — a function that takes a function and returns a wrapped version:

```python
def main_function(func):
    def wrapper():
        print("This is a sub function")
        func()
    return wrapper

@main_function     # same as: course_introduction = main_function(course_introduction)
def course_introduction():
    print("Please learn these concepts properly")

course_introduction()
```

### Decorators with arguments

Add one more layer: the outer function takes the decorator's argument and returns the actual
decorator. `*args, **kwargs` in the wrapper keep it usable on any function signature.

```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")     # prints three times
```

> **Gotcha:** return the inner function (`return wrapper`), **not** the call (`return wrapper()`).
> Returning the call executes it immediately at decoration time and binds `None` to the name.

---

## 9. Exception handling

> Notebooks: [01-exception-handling](01-python/04-exception-handling/01-exception-handling.ipynb) ·
> [02-custom-exceptions](01-python/04-exception-handling/02-custom-exceptions.ipynb)

`try` / `except` / `else` / `finally`:

```python
try:
    number = 10 / 0
except ZeroDivisionError:
    print("You cannot divide by zero.")

try:
    value = int("abc")
except ValueError:
    print("Invalid number input.")
```

- `try` — the code that might fail
- `except` — runs only if that specific exception was raised
- `else` — runs only if **no** exception was raised
- `finally` — runs either way (cleanup)

```python
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print("Division successful:", result)
finally:
    print("This runs no matter what.")
```

A file example, which is the usual real-world case:

```python
try:
    file = open("missing_file.txt", "r")
except FileNotFoundError:
    print("The file was not found.")
finally:
    print("File handling attempt finished.")
```

Catch the **specific** exception rather than a bare `except:` — otherwise real bugs get silently
swallowed alongside the one you meant to handle.

### Custom exceptions

Subclass `Exception` (usually via one base class per project) and `raise` it yourself:

```python
class Error(Exception):
    """Base class for this program's errors."""
    pass

class DOBException(Error):
    """Raised when the computed age is outside the allowed range."""
    pass

years = int(input("Enter your DOB: "))
age = 2024 - years

try:
    if 20 <= age <= 30:
        print("You are eligible for this program")
    else:
        raise DOBException()
except DOBException:
    print("Age is outside the eligible range.")
```

> Write the range check as `20 <= age <= 30`. Something like `20 >= age <= 30` reads as a range
> but actually means `20 >= age and age <= 30`, which is a different (and almost never intended)
> condition.

---

## 10. Modules, packages and the standard library

> Notebooks: [01-imports](01-python/05-modules-and-packages/01-imports.ipynb) ·
> [02-package-import](01-python/05-modules-and-packages/02-package-import.ipynb) ·
> [03-standard-library](01-python/05-modules-and-packages/03-standard-library.ipynb)

A **module** is a `.py` file. A **package** is a folder containing `__init__.py`.

```
05-modules-and-packages/
├── 01-imports.ipynb
└── package/
    ├── __init__.py      # marks the folder as a package; runs on import
    └── maths.py         # def addition(a, b): return a + b
```

```python
from package.maths import addition
print(addition(5, 3))    # 8
```

> `from package import addition` only works if `__init__.py` re-exports it, e.g. by adding
> `from .maths import addition` to it. Otherwise import the submodule explicitly, as above.

### Standard library tour

```python
import math
math.sqrt(16)      # 4.0
math.pi            # 3.14159...

import random
random.randint(1, 10)
random.choice(["apple", "banana", "cherry"])

import array
arr = array.array("i", [1, 2, 3, 4, 5])   # typed, compact array

import os
os.getcwd()
os.mkdir("test_dir")

import shutil
shutil.copy("a.txt", "b.txt")             # high-level file operations
```

**JSON — serialise and deserialise:**

```python
import json

data = {"name": "Alice", "age": 30}
json_str = json.dumps(data)        # dict -> JSON string
print(type(json_str))              # <class 'str'>

parsed = json.loads(json_str)      # JSON string -> dict
print(type(parsed))                # <class 'dict'>
```

Mnemonic: `dumps`/`loads` work on **strings**, `dump`/`load` work on **files**.

**CSV:**

```python
import csv

with open("example.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 30])

with open("example.csv", mode="r") as f:
    for row in csv.reader(f):
        print(row)
```

`newline=""` avoids blank rows between records on Windows.

**Dates and times:**

```python
from datetime import datetime, timedelta

now = datetime.now()
yesterday = now - timedelta(days=1)

import time
time.time()        # seconds since the epoch
time.sleep(1)
```

**Regular expressions:**

```python
import re

text = "There are 123 apples."
match = re.search(r"\d+", text)
print(match.group())    # 123
```

Use raw strings (`r"..."`) for patterns so backslashes reach the regex engine intact.

---

## 11. File handling and paths

> Notebooks: [01-file-operations](01-python/06-file-handling/01-file-operations.ipynb) ·
> [02-file-paths](01-python/06-file-handling/02-file-paths.ipynb)

Always use `with` — it closes the file even if an exception is raised.

### Modes

| Mode  | Meaning                                     |
|-------|---------------------------------------------|
| `r`   | read (file must exist)                      |
| `w`   | write — **truncates** an existing file      |
| `a`   | append                                      |
| `w+`  | write + read                                |
| `rb` / `wb` | binary read / write                   |

### Text files

```python
with open("../data/test.txt", "r") as file:
    content = file.read()          # whole file as one string

with open("../data/test.txt", "r") as file:
    for line in file:              # memory efficient - one line at a time
        print(line.strip())        # strip() removes the trailing newline

with open("../data/test.txt", "w") as file:   # overwrites!
    file.write("Some text to write to the file.")
    file.write("\nAnother line of text.")

with open("../data/test.txt", "a") as file:   # appends
    file.write("\nAnother line of text.")

lines = ["First line\n", "Second line\n", "Third line\n"]
with open("../data/test.txt", "a") as file:
    file.writelines(lines)         # note: writelines adds no newlines of its own
```

`seek(0)` rewinds the cursor, which is what makes `w+` read-after-write work:

```python
with open("../data/test.txt", "w+") as file:
    file.write("This is a sample text file.\n")
    file.seek(0)                   # back to the start
    print(file.read())
```

### Binary files

```python
data = b"\x00\x01\x02\x03\x04\x05"

with open("../data/test.bin", "wb") as file:
    file.write(data)

with open("../data/test.bin", "rb") as file:
    print(file.read())     # b'\x00\x01\x02\x03\x04\x05'
```

### A small utility — count lines, words, characters

```python
def count_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    line_count = len(lines)
    word_count = sum(len(line.split()) for line in lines)
    char_count = sum(len(line) for line in lines)
    return line_count, word_count, char_count

lines, words, characters = count_text_file("../data/test.txt")
print(f"Lines: {lines}, Words: {words}, Characters: {characters}")
```

### Paths with `os`

```python
import os

cwd = os.getcwd()
os.mkdir("Testdir")
os.listdir(cwd)

# join instead of string-concatenating - handles the / vs \ difference
os.path.join("Testdir", "testfile.txt")
os.path.join(cwd, "Testdir", "testfile.txt")

os.path.exists("test.txt")   # does it exist at all
os.path.isfile("test.txt")   # is it a file
os.path.isdir("Testdir")     # is it a directory
os.path.abspath("test.txt")  # relative -> absolute
```

Relative paths resolve against the **current working directory**, not the script's location —
which is why the notebooks in this repo reach their data as `../data/...`.

---

## 12. Logging

> Notebooks: [01-logging-basics](01-python/07-logging/01-logging-basics.ipynb) ·
> [02-multiple-loggers](01-python/07-logging/02-multiple-loggers.ipynb) ·
> script: [logging_demo.py](01-python/07-logging/logging_demo.py)

Logging replaces scattered `print()` calls with something you can filter, format and route.

### Levels

| Level      | When to use it                                          |
|------------|---------------------------------------------------------|
| `DEBUG`    | detailed diagnostic information                         |
| `INFO`     | confirmation that things are working as expected        |
| `WARNING`  | something unexpected, but the program still works       |
| `ERROR`    | a real failure — some function did not complete         |
| `CRITICAL` | a severe error; the program may not be able to continue |

Setting a level shows that level **and everything more severe**. The default is `WARNING`, which
is why `debug()` and `info()` look like they "do nothing" until you lower it.

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")
```

### Logging to a file, with a format

```python
import logging

logging.basicConfig(
    filename="app.log",
    filemode="w",                 # 'w' overwrites each run, 'a' appends
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

### Multiple loggers

Give each module its own named logger so you can set levels independently.

```python
import logging

logger1 = logging.getLogger("module1")
logger1.setLevel(logging.DEBUG)

logger2 = logging.getLogger("module2")
logger2.setLevel(logging.WARNING)     # module2 stays quiet below WARNING

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger1.debug("debug message from module1")     # shown
logger2.debug("debug message from module2")     # suppressed
logger2.warning("warning message from module2") # shown
```

The `%(name)s` field in the format string is what makes this useful — you can see which module
each line came from.

### Handlers — log to file *and* console at once

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app1.log"),   # to a file
        logging.StreamHandler(),           # to the console
    ],
)

logger = logging.getLogger("ArithmeticApp")

def divide(a, b):
    if b == 0:
        logger.error("Division by zero attempted")
        return None
    result = a / b
    logger.debug(f"Dividing {a} / {b}, result: {result}")
    return result
```

> `basicConfig` only takes effect the **first** time it runs in a process. In a notebook, restart
> the kernel if you change the configuration and nothing seems to change.

---

## 13. NumPy

> Notebook: [01-numpy](01-python/08-data-analysis/01-numpy.ipynb)

NumPy provides the `ndarray` — a fixed-type, contiguous N-dimensional array — plus fast
mathematical operations over it. It is the foundation everything else (Pandas, scikit-learn,
PyTorch, embedding vectors) is built on.

### Creating arrays

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])              # 1-D
arr3 = np.array([[1, 2, 3], [4, 5, 6]])       # 2-D

arr1.reshape(1, 5)          # change shape, same data
np.arange(0, 10, 2)         # [0 2 4 6 8]
np.ones((3, 4))             # 3x4 of ones
np.zeros((2, 2))            # 2x2 of zeros
np.eye(3)                   # 3x3 identity matrix
```

### Array attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape       # (2, 3)   rows, columns
arr.ndim        # 2        number of dimensions
arr.size        # 6        total elements
arr.dtype       # int64    element type
arr.itemsize    # 8        bytes per element
```

### Vectorised operations

Arithmetic applies element-wise across the whole array — no Python loop, and it runs in
compiled C.

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

arr1 + arr2    # [11 22 33 44 55]
arr1 - arr2    # [ -9 -18 -27 -36 -45]
arr1 * arr2    # [ 10  40  90 160 250]
arr1 / arr2    # [0.1 0.1 0.1 0.1 0.1]
```

### Universal functions (ufuncs)

```python
arr = np.array([1, 2, 3, 4, 5])

np.sqrt(arr)
np.exp(arr)
np.sin(arr)
np.log(arr)
```

### Indexing and slicing

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0][0]       # 1        - single element
arr[0, 0]       # 1        - same thing, preferred
arr[1:, 1:]     # [[5 6], [8 9]]   - rows from 1, columns from 1
arr[0:1, 1:]    # [[2 3]]

arr[0][0] = 100    # modify a single element
arr[1:] = 100      # broadcast 100 into every row from index 1
```

### Statistics and normalisation

```python
data = np.array([1, 2, 3, 4, 5])

np.mean(data)      # 3.0
np.var(data)       # 2.0
np.std(data)       # 1.414...
np.median(data)    # 3.0

# z-score normalisation: mean 0, standard deviation 1
normalized = (data - np.mean(data)) / np.std(data)
```

### Boolean masking

```python
data = np.array([1, 2, 3, 4, 5])

data > 2           # [False False  True  True  True]
data[data > 2]     # [3 4 5]  - filter using the mask
```

---

## 14. Pandas

> Notebooks: [02-pandas](01-python/08-data-analysis/02-pandas.ipynb) ·
> [03-reading-data-sources](01-python/08-data-analysis/03-reading-data-sources.ipynb) ·
> [04-data-manipulation](01-python/08-data-analysis/04-data-manipulation.ipynb)

Two structures: a **Series** (1-D labelled array) and a **DataFrame** (2-D labelled table —
think spreadsheet, or a dict of Series).

### Series

```python
import pandas as pd

series = pd.Series([1, 2, 3, 4, 5])
series = pd.Series({"a": 1, "b": 2, "c": 3})          # from a dict, keys become the index
series = pd.Series([10, 20, 30], index=[1, 2, 4])     # custom index
```

### DataFrame

```python
# from a dict of lists -> keys become COLUMNS
data = {
    "Name": ["Shaurya", "Rohit", "Ramesh", "Suresh"],
    "Age":  [25, 30, 35, 40],
    "City": ["New York", "Los Angeles", "Chicago", "Houston"],
}
df = pd.DataFrame(data, index=[1, 2, 3, 4])

# from a list of dicts -> each dict is a ROW
data = [
    {"Name": "Shaurya", "Age": 25, "City": "New York"},
    {"Name": "Rohit",   "Age": 30, "City": "Los Angeles"},
]
df = pd.DataFrame(data)

# from a file
df = pd.read_csv("../data/data.csv")
df.head(10)
```

### Accessing data

| Accessor            | Selects by         | Returns |
|---------------------|--------------------|---------|
| `df["Name"]`        | column name        | Series |
| `df.loc[1]`         | row **label**      | Series |
| `df.iloc[0]`        | row **position**   | Series |
| `df.at[1, "Age"]`   | label, label       | single value (fast) |
| `df.iat[2, 2]`      | position, position | single value (fast) |

The `loc` vs `iloc` distinction matters whenever the index isn't `0..n-1`.

### Adding, updating, dropping

```python
df["Salary"] = [50000, 60000, 70000, 80000]     # new column
df["Age"] = df["Age"] + 1                        # update a column
df.drop("Salary", axis=1, inplace=True)          # axis=1 -> column, axis=0 -> row
df.describe()                                    # count/mean/std/min/quartiles/max
```

> `inplace=True` mutates the DataFrame and returns `None`. Prefer `df = df.drop(...)` —
> it is clearer and chains.

### Reading from other sources

```python
import pandas as pd
from io import StringIO

# JSON
data = '{"employee_name": "James", "email": "james@example.com"}'
df = pd.read_json(StringIO(data))

df.to_json()                     # default: column-oriented
df.to_json(orient="index")
df.to_json(orient="records")     # list of row objects - the usual API shape

# CSV
df = pd.read_csv("file.csv", header=None)
df.to_csv("out.csv", index=False)

# HTML - scrapes every <table> on the page into a list of DataFrames
tables = pd.read_html("https://www.fdic.gov/bank-failures/failed-bank-list")
tables[0]

# Excel
df = pd.read_excel("file.xlsx", sheet_name="sheet1")
```

`StringIO` is needed because `read_json` expects a path or file-like object; wrapping the string
avoids the "passing literal json is deprecated" warning.

### Inspecting and cleaning

```python
df.head(5)         # first 5 rows
df.tail(5)         # last 5 rows
df.describe()      # summary statistics
df.dtypes          # column types

df.isnull().any()  # which columns have any missing values
df.isnull().sum()  # how many per column

df.fillna(0)                                             # fill with a constant
df["rev"] = df["Revenue"].fillna(df["Revenue"].mean())   # fill with the column mean

df.rename(columns={"rev": "main"}, inplace=True)
df["main"] = df["main"].astype(int)                      # change dtype
df["new"] = df["main"].apply(lambda x: x * 2)            # apply a function element-wise
```

### Grouping and aggregation

```python
df.groupby("Product")["main"].mean()

df.groupby("Category")["main"].agg(["mean", "sum", "count"])
```

Split → apply → combine: group the rows, run a function per group, stitch the results back.

### Merging

```python
df1 = pd.DataFrame({"key": ["A", "B", "C", "D", "E"], "value1": [1, 2, 3, 4, 5]})
df2 = pd.DataFrame({"key": ["A", "B", "C", "D"],      "value2": [5, 6, 7, 8]})

pd.merge(df1, df2, on="key", how="inner")   # only keys present in BOTH
pd.merge(df1, df2, on="key", how="outer")   # ALL keys; missing values become NaN
```

`how` also takes `left` and `right`. This is exactly a SQL join.

---

# Part 2 — GenAI / LangChain

## 15. Pydantic

> Notebook: [01-pydantic-basics](02-langchain/06-pydantic/01-pydantic-basics.ipynb)

Pydantic models use Python type annotations to **validate data at runtime**. LangChain, FastAPI
and structured LLM output all sit on top of it, which is why it comes before the LangChain work.

### Basic model

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

person = Person(name="John Doe", age=30, city="New York")
print(person)         # name='John Doe' age=30 city='New York'
print(type(person))   # <class '__main__.Person'>
```

Pass the wrong type and you get a `ValidationError` — this is the whole point. A plain dataclass
would happily accept `city=12`; Pydantic rejects it.

```python
Person(name="John Doe", age=30, city=12)   # ValidationError: city - input should be a valid string
```

### Optional fields and defaults

```python
from typing import Optional
from pydantic import BaseModel

class Employee(BaseModel):
    id: int
    name: str
    department: str
    salary: Optional[float] = None      # optional, defaults to None
    is_active: Optional[bool] = True    # optional, defaults to True

emp = Employee(id=1, name="Alice", department="HR")
print(emp)   # id=1 name='Alice' department='HR' salary=None is_active=True
```

`Optional[X]` means "an `X` or `None`". It still needs an explicit default to actually be
optional at construction time.

### Typed collections

```python
class Skill(BaseModel):
    skills: list[str]

Skill(skills=["Python", "Java", "C++"])
```

### Nested models

A model can use another model as a field type. A plain dict is coerced into the nested model
automatically.

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Customer(BaseModel):
    customer_id: int
    name: str
    address: Address

customer = Customer(
    customer_id=1,
    name="Emma",
    address={"street": "123 Main St", "city": "Boston", "zip_code": "02108"},
)

print(customer.address)        # Address(street='123 Main St', city='Boston', zip_code='02108')
print(customer.address.city)   # Boston
```

### `Field` — constraints and metadata

`Field` goes beyond type hints: value ranges, string lengths, defaults, aliases, descriptions.

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, le=1000)      # greater than 0, less than or equal to 1000
    quantity: int = Field(gt=0)
```

| Constraint | Meaning |
|------------|---------|
| `gt` / `ge` | greater than / greater than or equal |
| `lt` / `le` | less than / less than or equal |
| `min_length` / `max_length` | string or collection length |
| `default` | value used when the field is omitted |
| `description` | doc text — this is what an LLM sees when the model is used as a tool schema |

---

## 16. LangChain setup and LCEL chains

> Notebook: [01-getting-started](02-langchain/01-getting-started/01-getting-started.ipynb)

### Environment variables

Keys live in `.env` at the repo root and are loaded with `python-dotenv`. `load_dotenv()` walks
up from the working directory, so notebooks in any subfolder find the same file.

```python
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GEMINI_API_KEY"]    = os.getenv("GEMINI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_TRACING"] = "true"                       # turns on tracing
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
```

**LangSmith** is the observability layer. With `LANGSMITH_TRACING=true`, every chain run — the
prompt actually sent, the raw response, latency, token counts — shows up in the LangSmith UI.
Nothing else in the code changes.

### The model

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("What is langchain")
print(result.content)      # .content holds the text; the object also carries metadata
```

### Prompt templates

A `ChatPromptTemplate` separates the fixed instructions from the variable input, and `{input}`
is filled in at invoke time.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI engineer. Provide me with an answer with examples"),
    ("user", "{input}"),
])
```

- **system** — persona and standing rules
- **user** — the actual question

### Chains with LCEL (`|`)

The pipe operator wires components together. Output of the left becomes input of the right.

```python
chain = prompt | llm
res = chain.invoke({"input": "what is langsmith"})
print(res.content)
```

### Output parsers

Without a parser you get an `AIMessage` and have to reach for `.content`. `StrOutputParser`
makes the chain return a plain string.

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
res = chain.invoke({"input": "what is langsmith"})
print(res)      # already a str
```

**The core mental model:**

```
prompt template  ->  LLM  ->  output parser
   {input}          text        clean str
```

Everything later in this file is that same pipeline with a retriever bolted on the front.

---

## 17. Document loaders

> Notebook: [01-document-loaders](02-langchain/02-data-ingestion/01-document-loaders.ipynb)

A loader turns some source into a list of `Document` objects, each with `page_content` (the text)
and `metadata` (source, page number, title, …). Every RAG pipeline starts here.

### Text files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/speech.txt")
text_documents = loader.load()
```

### PDFs

`PyPDFLoader` returns **one Document per page**, with the page number in the metadata.

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("../data/sample.pdf")
pdf_documents = pdf_loader.load()
```

### Web pages

```python
from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader(web_paths=("https://docs.langchain.com/oss/python/langchain/overview",))
docs = web_loader.load()
```

`WebBaseLoader` uses BeautifulSoup under the hood — hence `beautifulsoup4` in requirements. It
takes `web_path=` (single) or `web_paths=` (a tuple of several).

### arXiv papers

```python
from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(query="1706.03762", load_max_docs=2)   # 1706.03762 = "Attention Is All You Need"
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])
print(docs[0].metadata)      # title, authors, published date, summary
```

The pattern is identical across all of them: **construct the loader, call `.load()`, get
`Document` objects back.** Swapping the source doesn't change anything downstream.

---

## 18. Text splitters

> Notebooks: [01-recursive-character-splitter](02-langchain/03-text-splitting/01-recursive-character-splitter.ipynb) ·
> [02-html-header-splitter](02-langchain/03-text-splitting/02-html-header-splitter.ipynb) ·
> [03-recursive-json-splitter](02-langchain/03-text-splitting/03-recursive-json-splitter.ipynb)

**Why split at all?** Embedding models have a token limit, and a whole document embedded as one
vector loses detail. Chunking gives the retriever small, specific pieces to match against.

### `RecursiveCharacterTextSplitter` — the default choice

It tries separators in order (`\n\n`, then `\n`, then ` `, then character) so paragraphs stay
intact where possible and are only broken up when a chunk would otherwise be too big.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_documents = PyPDFLoader("../data/attention.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50,     # characters repeated between neighbours
)

final_documents = text_splitter.split_documents(pdf_documents)
print(final_documents[0])
```

- **`chunk_size`** — the ceiling, in characters (not tokens).
- **`chunk_overlap`** — repeats the tail of one chunk at the head of the next so a sentence cut
  across the boundary still appears whole somewhere. Roughly 10–20% of `chunk_size` is typical.

> Use `split_documents()` on `Document` objects (metadata is preserved) and `create_documents()`
> on raw strings.

### `HTMLHeaderTextSplitter` — structure-aware

Splits at HTML element level and attaches the surrounding headers as metadata, so a chunk knows
which section it came from. That keeps related text together and keeps context attached.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_string)
```

### `RecursiveJsonSplitter` — for nested JSON

Traverses depth-first and builds smaller JSON objects, keeping nested structures whole where it
can and only splitting them to stay under the size limit.

```python
import requests
from langchain_text_splitters import RecursiveJsonSplitter

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=500)

json_chunks = json_splitter.split_json(json_data)       # -> list of dicts
docs = json_splitter.create_documents(texts=[json_data])  # -> list of Documents

for chunk in json_chunks[:3]:
    print(chunk)
```

Note: a very large **string value** inside the JSON will not be split — the splitter works on the
object structure, not on text inside a leaf.

### Choosing a splitter

| Source | Splitter |
|--------|----------|
| Plain text, PDFs, most things | `RecursiveCharacterTextSplitter` |
| HTML where sections matter | `HTMLHeaderTextSplitter` |
| Nested JSON / API schemas | `RecursiveJsonSplitter` |
| Fixed single separator | `CharacterTextSplitter` |

---

## 19. Embeddings

> Notebooks: [01-huggingface-embeddings](02-langchain/04-embeddings/01-huggingface-embeddings.ipynb) ·
> [02-ollama-embeddings](02-langchain/04-embeddings/02-ollama-embeddings.ipynb)

An embedding maps text to a dense vector of floats. Texts with similar meaning land close
together, which is what makes semantic search work — the retriever compares vectors, not keywords.

### HuggingFace (local, runs on your machine)

```python
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectors = embeddings.embed_documents([
    "Alpha is the first letter of the Greek alphabet."
])
```

`all-MiniLM-L6-v2` is small (384 dimensions), fast, and downloads once via
`sentence-transformers`.

### Ollama (local server)

```python
from langchain_ollama import OllamaEmbeddings

# gemma:2b is a CHAT model - it has no embedding head, so Ollama returns
# HTTP 500 "This server does not support embeddings".
# Use a dedicated embedding model instead: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectors = embeddings.embed_documents([
    "Alpha is the first letter of the Greek alphabet."
])
```

That comment is a real bug hit during this work — **chat models and embedding models are not
interchangeable.** Pull an embedding model explicitly.

### Google Gemini (API)

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
```

### The two methods

| Method | Use for |
|--------|---------|
| `embed_documents(list_of_texts)` | the corpus being indexed |
| `embed_query(single_text)` | the incoming search query |

They are separate because some models prepend different instruction prefixes for documents vs
queries. **Index and query must use the same embedding model** — vectors from different models
are not comparable.

---

## 20. Vector stores

> Notebooks: [01-chroma](02-langchain/05-vector-stores/01-chroma.ipynb) ·
> [02-faiss](02-langchain/05-vector-stores/02-faiss.ipynb)

A vector store holds the embeddings and answers "which chunks are closest to this query".

### Chroma

An AI-native open-source vector database (Apache 2.0), aimed at developer ergonomics.

```python
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../data/speech.txt").load()
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30).split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(docs, embeddings)

results = db.similarity_search(query)
print(results[0].page_content)

db.similarity_search_with_score(query)     # (Document, score) pairs
```

### FAISS

Facebook AI Similarity Search — a library for efficient similarity search over dense vectors,
scaling to sets that don't fit in RAM.

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

documents = TextLoader("../data/speech.txt").load()
docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30).split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embeddings)

results = db.similarity_search(query)
print(results[0].page_content)
```

**Similarity search with score.** FAISS returns **L2 distance**, so a *lower* score is better —
the opposite of a similarity score, and easy to misread.

```python
docs_and_scores = db.similarity_search_with_score(query)
```

**Saving and reloading** so you don't re-embed every run:

```python
db.save_local("faiss_index")

new_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,   # the index is loaded with pickle
)
```

`allow_dangerous_deserialization=True` is required because loading unpickles a file — only pass
it for an index you created yourself.

### As a retriever

Converting the store into a `Retriever` gives it the standard interface the rest of LangChain
expects, so it can be dropped into any chain.

```python
retriever = db.as_retriever()
retriever.invoke(query)
```

### Chroma vs FAISS

| | Chroma | FAISS |
|---|--------|-------|
| Type | database | library |
| Persistence | built in (`persist_directory`) | `save_local` / `load_local` |
| Metadata filtering | rich | limited |
| Scale | good | very large sets, highly tuned |

---

## 21. Building a RAG pipeline

> Notebook: [02-simple-rag-pipeline](02-langchain/01-getting-started/02-simple-rag-pipeline.ipynb)

This ties everything above into one flow.

```
Load  ->  Split  ->  Embed  ->  Store  ->  Retrieve  ->  Stuff into prompt  ->  LLM  ->  Answer
```

### 1. Load and split

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = WebBaseLoader(web_path="https://docs.langchain.com/langsmith/evaluation-quickstart").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
text_chunks = text_splitter.split_documents(docs)
```

### 2. Embed and store

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
db = Chroma.from_documents(text_chunks, embeddings)

result = db.similarity_search(query)
result[0].page_content
```

### 3. The document chain (a.k.a. "stuff" chain)

`create_stuff_documents_chain` takes retrieved documents, **stuffs** them all into the `{context}`
slot of the prompt, and sends the whole thing to the LLM.

The "if the answer is not in the text, say I don't know" instruction is what keeps the model
grounded in the retrieved context instead of falling back on its own memory.

```python
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_template("""
Answer the following based on the context below, and if the answer is not contained
within the text below, say "I don't know"

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
```

Called directly, you have to supply the context yourself:

```python
document_chain.invoke({
    "input": question,
    "context": db.similarity_search(question),
})
```

### 4. The retrieval chain

`create_retrieval_chain` wires the retriever in front, so the context is fetched automatically.
This is the piece that makes it *retrieval-augmented*.

```python
from langchain_classic.chains import create_retrieval_chain

retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

res = retrieval_chain.invoke({"input": "How are LLM applications evaluated?"})

res["answer"]     # the generated answer
res["context"]    # the chunks that were actually retrieved and used
res["input"]      # the original question
```

`res["context"]` is the debugging handle: if an answer is wrong, look there first. Bad answers
are usually a **retrieval** problem (wrong chunks fetched), not a generation problem.

---

## 22. Serving it with Streamlit

> Script: [streamlit_ollama_app.py](02-langchain/apps/streamlit_ollama_app.py)

The same prompt → LLM → parser chain, wrapped in a web UI, running fully locally against Ollama.

```python
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked"),
    ("user", "{question}"),
])

st.title("Langchain with gemma2")
input_text = st.text_input("Enter your question here")

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write("Response: ", response)
```

Run it with:

```bash
cd 02-langchain/apps
streamlit run streamlit_ollama_app.py
```

Streamlit re-runs the whole script top to bottom on every interaction — that's why the `if
input_text:` guard is needed, and why expensive setup normally goes behind `@st.cache_resource`.

---

## 23. Gotchas worth remembering

Things that actually cost time during this work:

1. **Chat models ≠ embedding models.** `OllamaEmbeddings(model="gemma:2b")` returns
   HTTP 500 "This server does not support embeddings". Pull a real embedding model:
   `ollama pull nomic-embed-text`.

2. **`pip install fitz` is the wrong package.** The `fitz` module comes from **PyMuPDF**.
   The PyPI project literally named `fitz` is unrelated and will break the import.

3. **FAISS scores are distances.** `similarity_search_with_score` returns L2 distance, so
   **lower is better**. Don't sort it as if it were a similarity.

4. **`FAISS.load_local` needs `allow_dangerous_deserialization=True`** because it unpickles.
   Only do this for indexes you created yourself.

5. **Never name a file after a stdlib module.** A `logging.py` in the working directory shadows
   the real `logging` module and produces baffling import errors. (An orphaned
   `__pycache__/logging.cpython-312.pyc` in the old folder was evidence of exactly this.)

6. **`basicConfig` is once-per-process.** Change the logging config in a notebook and nothing
   happens until you restart the kernel.

7. **`20 >= age <= 30` is not a range check.** It chains to `20 >= age and age <= 30`. Write
   `20 <= age <= 30`.

8. **Decorators must return the function, not call it.** `return wrapper`, never `return wrapper()`.

9. **`{}` is an empty dict, not an empty set.** Use `set()`.

10. **`(5)` is an int; `(5,)` is a tuple.** The comma makes the tuple, not the parentheses.

11. **Relative paths resolve against the working directory**, not the notebook's folder — which
    is why data is reached as `../data/...` here.

12. **Index and query with the same embedding model.** Vectors from different models are not
    comparable, and the failure is silent — you just get bad results.

---

## Where to go next

- **Chat history / memory** — multi-turn conversations over the RAG chain
- **Agents and tools** — letting the model choose which tool to call
- **Structured output** — the Pydantic models from §15 as LLM response schemas
- **Retrieval quality** — MMR, metadata filtering, hybrid search, re-ranking
- **Evaluation** — LangSmith datasets and evaluators over the pipeline
