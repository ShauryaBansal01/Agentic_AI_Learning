# Agentic AI Learning — Notes

Everything covered so far, in the order it was learned: **plain Python → advanced Python →
data libraries → Pydantic → LangChain → RAG → LangChain v1 agents → LangGraph**. Every section links to the notebook it came
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
23. [Groq and the LCEL translation chain](#23-groq-and-the-lcel-translation-chain)
24. [Serving a chain with LangServe](#24-serving-a-chain-with-langserve)
25. [Chatbots with message history](#25-chatbots-with-message-history)
26. [Documents, retrievers and a hand-built RAG chain](#26-documents-retrievers-and-a-hand-built-rag-chain)

**Part 3 — LangChain v1**
27. [What changed in v1, and the `uv` project](#27-what-changed-in-v1-and-the-uv-project)
28. [Agents with `create_agent`](#28-agents-with-create_agent)
29. [`init_chat_model` and provider strings](#29-init_chat_model-and-provider-strings)
30. [Messages in v1](#30-messages-in-v1)
31. [Tools and the tool-execution loop](#31-tools-and-the-tool-execution-loop)
32. [Structured output](#32-structured-output)
33. [Agent middleware](#33-agent-middleware)

**Part 4 — LangGraph**
34. [Building a graph from scratch](#34-building-a-graph-from-scratch)

**Reference**
35. [Gotchas worth remembering](#35-gotchas-worth-remembering)

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

## 23. Groq and the LCEL translation chain

> Notebook: [SimplellmLCEL](02-langchain/07-LCEL/SimplellmLCEL.ipynb)

Everything up to here ran either on Gemini (API) or Ollama (local). **Groq** is a third option: a
hosted inference provider that serves *open-weight* models on custom hardware, so it is fast and
has a generous free tier. The LangChain surface is identical — only the constructor changes.

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)
```

Add `GROQ_API_KEY` to `.env` (key from <https://console.groq.com/keys>).

### Messages instead of a prompt template

Section 16 built the prompt from a `ChatPromptTemplate`. A model can also be invoked with a plain
**list of message objects** — useful when the conversation is assembled in code rather than from a
template.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content="Hello, how are you?"),
]

res = model.invoke(messages)
res.content        # 'Bonjour, comment ça va?'
```

`res` is an `AIMessage`. `StrOutputParser` pulls the text out of it:

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(res)      # 'Bonjour, comment ça va ?'
```

### Chaining without a prompt

Any two runnables compose, so a chain does not have to start with a prompt. Here the chain input
*is* the message list:

```python
chain = model | output_parser
chain.invoke(messages)         # 'Bonjour, comment allez-vous ?'
```

### Adding the template back

Hard-coding "to French" in the system message means a new message list for every language. A
template makes both the language and the text parameters:

```python
from langchain_core.prompts import ChatPromptTemplate

generic_template = "Translate the following from English to {language}"

prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}"),
])

prompt.invoke({"language": "Japanese", "text": "Hello How are you ?"})
# ChatPromptValue(messages=[SystemMessage(...), HumanMessage(...)])
```

`prompt.invoke(...)` returns a `ChatPromptValue` — the rendered messages, *before* the model sees
them. Handy for checking what actually gets sent.

The full three-stage chain:

```python
chain = prompt | model | output_parser
chain.invoke({"language": "Japanese", "text": "Hello, how are you?"})
# 'こんにちは、お元気ですか？'
```

```
dict input   ->   prompt    ->   model     ->  parser  ->  str
{language,        messages       AIMessage      'こんにちは…'
 text}
```

That chain is reused as-is in the next two sections.

---

## 24. Serving a chain with LangServe

> Script: [langserver.py](02-langchain/apps/langserver.py)

Streamlit (§22) gives a chain a *UI*. **LangServe** gives it an *HTTP API* — it takes any runnable
and mounts it onto a FastAPI app as a set of REST endpoints, with request and response schemas
derived from the chain itself.

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="LangChain LangServer",
    description="A simple API server using Langchain runnable interfaces",
    version="1.0",
)

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

`chain` here is exactly the `prompt | model | output_parser` from §23.

Run it:

```bash
cd 02-langchain/apps
python langserver.py
```

### What `add_routes` actually mounts

One call generates the whole surface (checked against the running app):

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/chain/invoke` | POST | one input, one output |
| `/chain/batch` | POST | a list of inputs in a single request |
| `/chain/stream` | POST | server-sent events, token by token |
| `/chain/stream_log`, `/chain/stream_events` | POST | intermediate steps as they run |
| `/chain/input_schema`, `/chain/output_schema`, `/chain/config_schema` | GET | JSON Schema, generated from the runnable |
| `/chain/playground/` | GET | built-in browser UI for poking at the chain |
| `/docs` | GET | FastAPI's own Swagger UI |

Calling it:

```bash
curl -X POST http://127.0.0.1:8000/chain/invoke \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"language\": \"Japanese\", \"text\": \"Hello, how are you?\"}}"
```

The chain's own input goes **inside** an `"input"` key, and the answer comes back under
`"output"`. That envelope is LangServe's convention, not the chain's.

The takeaway: the chain is written and tested once in a notebook, then served without rewriting any
of it. Streamlit and LangServe are two different front doors onto the same runnable.

---

## 25. Chatbots with message history

> Notebook: [1-chat-bot](02-langchain/chat-bot/1-chat-bot.ipynb)

### The problem: models are stateless

Each `invoke` is an independent HTTP request. The model remembers nothing between them.

```python
model.invoke([HumanMessage(content="Hello my name is shaurya, I'm chief AI Engineer")])
# "Hello Shaurya! It's great to meet the Chief AI Engineer..."

model.invoke([HumanMessage(content="Hey what's my name and what do I do")])
# no idea - that was a separate request
```

Memory is not a model feature; it is *replaying the transcript* on every call. Done by hand, it
works:

```python
from langchain_core.messages import AIMessage

model.invoke([
    HumanMessage(content="Hello my name is shaurya, I'm chief AI Engineer"),
    AIMessage(content="Hello shaurya, nice to meet you! How can I assist you today?"),
    HumanMessage(content="Hey what's my name and what do I do"),
])
# "You're **Shaurya**, and you're a **Chief AI Engineer**."
```

### `RunnableWithMessageHistory` — automating the replay

Rather than rebuilding that list by hand, wrap the runnable. It appends each input and output to a
store keyed by **session id**, and prepends the history on the next call.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

store = {}                      # session_id -> history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
```

The session id is passed at invoke time, under `configurable`:

```python
config = {"configurable": {"session_id": "chat1"}}

with_message_history.invoke(
    [HumanMessage(content="Hello my name is shaurya, I'm chief AI Engineer")],
    config=config,
)

with_message_history.invoke(
    [HumanMessage(content="what is my name and what do i do ?")],
    config=config,
).content
# "You're Shaurya, and you're a Chief AI Engineer..."
```

Change `session_id` to `"chat2"` and it is a fresh conversation — that is how one process serves
many users. `ChatMessageHistory` is **in-memory**: `store` is a plain dict, so everything is lost
when the kernel restarts. Swap in a persistent history class for anything real.

### Adding a system prompt: `MessagesPlaceholder`

To put a persona in front of the conversation, the template needs a slot where the *whole history*
is spliced in — that is `MessagesPlaceholder`, as opposed to `{input}`, which takes one string.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant, answer all the questions to the best of your abilities"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model
chain.invoke({"messages": [HumanMessage(content="Hello my name is shaurya")]})
```

### `input_messages_key` — the part that bites

Wrapping the *model* works with a bare list, because the entire input is the messages. Wrapping the
*chain* does not: the chain takes a **dict**, so the wrapper has to be told which key holds the
messages.

```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",      # matches MessagesPlaceholder(variable_name="messages")
)

config = {"configurable": {"session_id": "chat2"}}

with_message_history.invoke(
    {"messages": [HumanMessage(content="Hello my name is shaurya, I'm chief AI Engineer")]},
    config=config,
).content
```

The string in `input_messages_key` must match the `variable_name` of the placeholder. Leave it out
on a dict-input chain and the history has nowhere to go.

```
              +---------- get_session_history(session_id) ----------+
              |                                                     |
user input -> RunnableWithMessageHistory -> prompt -> model -> AIMessage
              |      (reads history, then appends both sides)       |
              +-----------------------------------------------------+
```

### Trimming the history so it fits the context window

`store` only ever grows. Every turn replays the whole transcript, so a long conversation eventually
overflows the model's context window — and pays for the whole thing again on every call.
`trim_messages` is a runnable that cuts the list down *before* it reaches the model.

```python
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",               # keep the END of the conversation
    token_counter="approximate",   # or pass the model itself: token_counter=model
    include_system=True,           # never drop the system prompt
    allow_partial=False,           # don't cut a message in half
    start_on="human",              # the surviving window must begin with a HumanMessage
)

messages = [
    SystemMessage(content="you are a helpful assistant, answer all the questions to the best of your abilities"),
    HumanMessage(content="Hello my name is shaurya, I'm chief AI Engineer"),
    AIMessage(content="Hello shaurya, nice to meet you! How can I assist you today?"),
    HumanMessage(content="Hey what's my name and what do I do"),
]

trimmer.invoke(messages)
# [SystemMessage('you are a helpful assistant...'),
#  HumanMessage("Hey what's my name and what do I do")]
```

| Argument | What it controls |
| --- | --- |
| `max_tokens` | the budget the surviving messages must fit into |
| `strategy` | `"last"` keeps the most recent turns, `"first"` keeps the oldest |
| `token_counter` | `"approximate"` for a cheap estimate, or a chat model for its real tokeniser |
| `include_system` | keeps the `SystemMessage` regardless of the budget |
| `allow_partial` | whether a message may be truncated mid-way to fit |
| `start_on` | message type the trimmed list must start with — keeps human/AI turns paired |

### Wiring the trimmer into the chain

The trimmer runs on the `messages` key, and everything else about the chain stays the same:

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
```

`RunnablePassthrough.assign(...)` passes the input dict straight through but **overwrites one key**
— here `messages` is replaced by its trimmed version. `itemgetter("messages")` pulls that key out
of the dict, so the trimmer receives a plain list rather than the dict.

### Trimming is lossy — that is the point

```python
chain.invoke({"messages": messages + [HumanMessage(content="Hey what do I do and what is my name")]})
# "I don't have any personal info about you, so I can't tell you your name..."
```

At 45 tokens the trimmer kept only the system message and the final question — the turn where the
name was introduced fell outside the budget, so the model genuinely cannot answer. Nothing errored;
the bot simply forgot. The budget *is* the memory span, and the same thing happens after wrapping
the trimmed chain in `RunnableWithMessageHistory`:

```python
with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)
with_message_history.invoke({"messages": messages + [...]}, config={"configurable": {"session_id": "chat3"}})
```

The history keeps accumulating in `store`; the trimmer just narrows the slice of it that the model
actually sees.

---

## 26. Documents, retrievers and a hand-built RAG chain

> Notebook: [vectorretriver](02-langchain/vectorrectriver/vectorretriver.ipynb)

§21 built a RAG pipeline out of the prebuilt `create_stuff_documents_chain` /
`create_retrieval_chain` helpers. This section takes the lid off and assembles the same thing from
plain runnables, which is what those helpers are doing underneath.

### The `Document` abstraction

Every loader, splitter and vector store in LangChain speaks in `Document` objects — one unit of
text plus arbitrary metadata:

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="...text of the chunk...", metadata={"source": "doc1.txt"}),
    Document(page_content="...", metadata={"source": "doc2.txt"}),
    Document(page_content="...", metadata={"source": "doc3.txt"}),
]
```

- `page_content` — a string, the text itself.
- `metadata` — a dict: where it came from, what it relates to, anything worth filtering on later.

A `Document` is usually a **chunk** of a larger file, not the whole file — that is what §18's
splitters produce. Writing them by hand, as above, is just a way to get a controlled corpus for
experimenting.

### Store: Chroma over HuggingFace embeddings

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding=embeddings)
```

`all-MiniLM-L6-v2` runs locally, downloads once (~90 MB) and needs no server — handy when Ollama
isn't running. Without a `persist_directory` the collection lives in memory and disappears with the
kernel.

Three ways to query it:

```python
# sync
vectorstore.similarity_search("knmbdhjiabedhbjad", k=2)

# async — same call inside an event loop (a notebook cell can `await` directly)
await vectorstore.asimilarity_search("knmbdhjiabedhbjad", k=2)

# with scores
vectorstore.similarity_search_with_score("knmbdhjiabedhbjad", k=2)
# [(Document(...), 2.43e-13), ...]
```

Every LangChain store exposes the `a`-prefixed async twin of each method. The score is a **distance
again, not a similarity** — the near-zero value above is a query that was an exact copy of a stored
document.

### Why retrievers exist

A `VectorStore` is **not** a `Runnable`, so it cannot be dropped into an LCEL chain — no `|`, no
`.batch()`, no async interface for free. A `Retriever` *is* a runnable wrapper around one search
method of the store.

Built by hand, to show there is no magic in it:

```python
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
retriever.batch(["knmbdhjiabedhbjad"])      # -> [[Document(...)]]
```

`RunnableLambda` turns any callable into a runnable; `.bind(k=1)` freezes an argument so the chain
only has to supply the query. `batch` then comes for free.

The built-in version does the same thing with the knobs named:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
retriever.batch(["knmbdhjiabedhbjad"])
```

| `search_type` | What it does |
| --- | --- |
| `"similarity"` | plain nearest-neighbour, the default |
| `"mmr"` | maximal marginal relevance — trades a little relevance for less redundancy |
| `"similarity_score_threshold"` | drops anything below `score_threshold`, so it may return nothing |

`search_kwargs` is passed straight through to the underlying store call (`k`, `filter`,
`score_threshold`, …).

### The RAG chain, without the helpers

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

rag_chain.invoke("knmbdhjiabedhbjad").content
```

The dict is the whole trick. A plain `dict` in an LCEL chain is coerced into a **`RunnableParallel`**:
each value receives the *same* input and runs, and the results are collected into a dict with the
same keys. So the single string `"knmbdhjiabedhbjad"` goes to both branches at once —
`retriever` turns it into documents for `{context}`, while `RunnablePassthrough()` hands it
through unchanged for `{question}`.

```
                 +-- retriever ------------> context --+
"question str" --+                                     +--> prompt -> llm -> AIMessage
                 +-- RunnablePassthrough --> question -+
```

That is exactly what `create_retrieval_chain` wraps up. Worth knowing both: the helper for real
work, the manual form for when the shape of the chain has to change.

---

# Part 3 — LangChain v1

## 27. What changed in v1, and the `uv` project

> Project: [Langchainupdated/](Langchainupdated/)

Everything in Part 2 was written against LangChain 0.x. **LangChain 1.x is a different API** — not
a coat of paint. Rather than rewrite the old notebooks, the v1 work lives in its own project —
[Langchainupdated/](Langchainupdated/), a `uv` project on Python 3.11 with its own `.env` — so both
versions stay runnable side by side.

```bash
cd Langchainupdated
uv sync                 # reads pyproject.toml + uv.lock, builds .venv
```

`uv` replaces `pip install -r requirements.txt` here: `pyproject.toml` declares the dependencies,
`uv.lock` pins the exact resolved versions, and `.python-version` pins the interpreter. `uv sync`
reproduces the whole environment from those three files.

What actually changed in v1, in one table:

| 0.x (Part 2) | 1.x (Part 3) |
| --- | --- |
| `ChatGroq(...)`, `ChatGoogleGenerativeAI(...)` | `init_chat_model("groq:openai/gpt-oss-120b")` |
| `from langchain_core.messages import ...` | `from langchain.messages import ...` |
| `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| chains stitched by hand with `\|` | `create_agent(...)` — a compiled LangGraph |
| `res.content` is always a `str` | `res.content` may be a list of **content blocks**; `res.text` is the string |
| `create_stuff_documents_chain`, `create_retrieval_chain` | moved to `langchain-classic` |

LCEL still works — `prompt | model | parser` is unchanged. What v1 adds is a level above it.

---

## 28. Agents with `create_agent`

> Notebook: [langchain.ipynb](Langchainupdated/updatedlangchain/langchain.ipynb)

Every chain up to here was a fixed pipeline: the steps and their order were decided when the chain
was written. An **agent** inverts that — the model decides which tools to call, with what
arguments, and how many times, before it answers.

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get the weather of this city."""
    return f"The weather in {city} is sunny"

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent      # <langgraph.graph.state.CompiledStateGraph object at 0x...>
```

Note the type: `create_agent` returns a **`CompiledStateGraph`** — a LangGraph graph, not a
`Runnable` chain. That is the actual headline of v1: agents are graphs with a loop in them
(model → tool → model → …), which a straight-line `|` pipeline cannot express.

A plain function is enough to be a tool here. The **docstring is not decoration** — it is the
description the model reads when deciding whether to call it, and the type hints become the
argument schema.

### Invoking it

The input is a state dict, and `messages` is the channel the conversation lives on:

```python
res = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in delhi"}]})

res["messages"][-1].content      # 'The weather in delhi is sunny'
```

`res["messages"]` is the whole transcript, not just the answer — the human turn, the `AIMessage`
carrying the tool call, the `ToolMessage` with the result, and the final `AIMessage`. The last
element is the answer; the rest is the trace of how it got there.

```
{"messages": [...]}  ->  model  --tool_calls-->  tool  --ToolMessage-->  model  ->  final answer
                           ^                                                |
                           +--------------- loops until no tool call -------+
```

A bare string in place of the list also works — `agent.invoke({"messages": "what is the weather in
delhi"})` — but the explicit `{"role", "content"}` form is what scales to multi-turn.

---

## 29. `init_chat_model` and provider strings

> Notebook: [modelintegration.ipynb](Langchainupdated/updatedlangchain/modelintegration.ipynb)

Part 2 imported a different class per provider. v1 adds one factory that takes a
`"provider:model"` string, so swapping providers is a one-string edit:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-flash-latest")
model = init_chat_model("groq:openai/gpt-oss-20b")
```

The provider-specific classes still exist and behave identically — `init_chat_model` just picks
one for you:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
model = ChatGroq(model="openai/gpt-oss-120b")
```

The provider package still has to be installed — `init_chat_model` resolves the string to a class,
it does not vendor the SDK.

### `.content` vs `.text`

A v1 response's `content` is no longer guaranteed to be a string. For Gemini it comes back as a
list of **content blocks**:

```python
res = model.invoke("Hi how are you")
res.content
# [{'type': 'text', 'text': "Hello! I'm doing well...", 'extras': {'signature': '...'}}]
```

That is the shape that lets one message carry text *and* reasoning *and* tool calls together.
`res.text` always gives the plain string, so prefer it over `res.content` when all you want is the
answer.

### Streaming

```python
for chunk in model.stream("Write me a 500 word paragraph on Artificial Intelligence"):
    print(chunk.text, end="|", flush=True)
```

Each chunk is a partial `AIMessage`. The run of empty `|` separators at the start is real: the
model is emitting reasoning-token chunks that carry no text yet.

### Batch

`batch` sends several independent prompts and returns the answers in the same order:

```python
responses = model.batch([
    "why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "what is quantum computing?",
])
```

They run **concurrently**, which is the point — three sequential `invoke` calls would take three
round-trips. Cap the fan-out when the provider rate-limits:

```python
model.batch([...], config={"max_concurrency": 5})
```

`invoke` / `stream` / `batch` (and their `a`-prefixed async twins) are the standard runnable
interface — every chain in Part 2 has them too.

---

## 30. Messages in v1

> Notebook: [mesaages.ipynb](Langchainupdated/updatedlangchain/mesaages.ipynb)

The import path moved from `langchain_core.messages` to `langchain.messages`; the objects are the
same ones from §23.

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage
```

| Type | What it is |
| --- | --- |
| `SystemMessage` | context, persona, instructions — how the model should behave |
| `HumanMessage` | user input |
| `AIMessage` | what the model generated |
| `ToolMessage` | the result of a tool call, fed back to the model |

**A plain string is a valid prompt** — `model.invoke("what is langchain")` — and it is the right
choice for one-shot generation where no history is being kept. The list-of-messages form is for
when the roles matter:

```python
messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a poem of AI/ML"),
]
res = model.invoke(messages)
res.content
```

Note the positional form: `SystemMessage("...")` works as well as `SystemMessage(content="...")`.

Messages also carry optional identity metadata, which is what makes multi-user transcripts
readable later:

```python
human_msg = HumanMessage(content="hello", name="Shaurya", id="123")
model.invoke([human_msg])
```

`name` and `id` travel with the message through the whole graph; not every provider forwards
`name` to the model, but it survives in the transcript either way.

---

## 31. Tools and the tool-execution loop

> Notebook: [tools.ipynb](Langchainupdated/updatedlangchain/tools.ipynb)

`create_agent` (§28) runs the tool loop for you. This section runs it by hand, which is the only
way to see what the agent is actually doing.

A tool is two things bolted together:

1. a **schema** — name, description, and argument definitions (JSON Schema under the hood), and
2. a **function** to execute when the model asks for it.

The `@tool` decorator builds both from the Python function:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather of the location"""
    return f"its sunny in {location}"
```

The name comes from the function, the argument schema from the type hints, and the description
**from the docstring** — that is the text the model reads when choosing. A vague docstring is a
tool the model calls at the wrong moment.

### Binding tools to a model

```python
model_with_tools = model.bind_tools([get_weather])
```

`bind_tools` returns a *new* model with the tool schemas attached to every request; the original is
untouched. Binding does not give the model the ability to *run* anything — it can only ask.

```python
res = model_with_tools.invoke("Whats the weather in delhi")

res.content        # '' - nothing to say yet
res.tool_calls     # [{'name': 'get_weather', 'args': {'location': 'Delhi'}, 'id': 'fc_...', ...}]

for tool_call in res.tool_calls:
    print(tool_call["name"], tool_call["args"])
```

An `AIMessage` with tool calls usually has **empty content**. It is a request, not an answer — and
that is the whole trap: whatever runs the loop has to check `tool_calls`, not just print `content`.

### The three-step loop

```python
# 1. the model generates tool calls
message = [{"role": "user", "content": "What is the weather in delhi"}]
ai_msg = model_with_tools.invoke(message)
message.append(ai_msg)

# 2. execute each call and collect the results
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)     # -> ToolMessage
    message.append(tool_result)

# 3. hand the results back so the model can answer
final_res = model_with_tools.invoke(message)
final_res.text          # 'The current weather in Delhi is sunny. Enjoy the clear skies!'
```

Passing the **whole `tool_call` dict** to `.invoke()` — not just `tool_call["args"]` — is what makes
the tool return a `ToolMessage` already stamped with the matching `tool_call_id`. Pass only the args
and you get the bare return value, leaving you to build the `ToolMessage` and pair the id yourself.
Get that pairing wrong and the provider rejects the next request.

The accumulated `message` list tells the story:

```
[ {'role': 'user', ...},                       the question
  AIMessage(content='', tool_calls=[...]),     the model asking for a tool
  ToolMessage('its sunny in Delhi', ...),      what the tool returned
  AIMessage('The current weather in Delhi...') the answer built from it ]
```

That is exactly the transcript `create_agent` produces in `res["messages"]` — the agent *is* this
loop, wrapped in a graph that repeats it until the model stops asking for tools.

---

## 32. Structured output

> Notebook: [structuredOutput.ipynb](Langchainupdated/updatedlangchain/structuredOutput.ipynb)

Prose is fine for a human reader and useless for the next line of code. `with_structured_output`
binds a schema to the model so `invoke` returns a **typed object** instead of a string — no regexes,
no JSON parsing, no "sometimes it adds a preamble".

### Pydantic — the richest option

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The movie was released this year")
    director: str = Field(description="The director of this movie")
    rating: float = Field(description="The movies rating out of 10")

structure_model = model.with_structured_output(Movie)
structure_model.invoke("Provide Details about the movie Inception")
# Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)
```

Every `description` is sent to the model as part of the schema, so they are prompt text, not
comments. The **field names are prompt text too** — `rating` is understood, a typo like `bugdet` is
not, and the model is left guessing what was meant. Spell the schema the way it should be read.

Under the hood this is tool calling: the model "calls" a function named after the schema, and
LangChain validates the arguments back into the model class. §15 wrote these Pydantic models as
plain validation; this is the payoff.

### Getting the raw message too

```python
structure_model = model.with_structured_output(Movie, include_raw=True)
res = structure_model.invoke("Provide Details about the movie inception")

res["parsed"]          # Movie(...)
res["raw"]             # the AIMessage, with token usage and the underlying tool call
res["parsing_error"]   # None if it validated
```

`include_raw=True` changes the return type from the model instance to a **dict** — worth knowing
before it breaks the line after it. It is also what makes parse failures recoverable: without it, a
schema violation raises.

### Nesting

Schemas compose, so one call can fill in a whole object graph:

```python
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model.with_structured_output(MovieDetails).invoke("Provide Details about the movie inception")
# MovieDetails(title='Inception', year=2010,
#              cast=[Actor(name='Leonardo DiCaprio', role='Cobb'), ...],
#              genres=['Science Fiction', 'Action', 'Thriller', 'Heist'], budget=None)
```

`budget=None` is the schema working as intended: the field is optional, the model did not know, so
it left it out instead of inventing a number.

### `TypedDict` — no runtime validation

When a plain dict is enough and validation is not needed:

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "Year of release"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "Rating out of 10"]

model.with_structured_output(MovieDict).invoke("Provide Details about the movie inception")
# {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

`Annotated[type, ..., "description"]` is how a `TypedDict` carries descriptions. A `TypedDict`
**cannot take defaults**, so `budget: float | None = Field(None, description=...)` in the class body
does nothing — the `Field` is ignored and only the annotation reaches the model.

| Schema type | Returns | Validation | Descriptions via |
| --- | --- | --- | --- |
| Pydantic `BaseModel` | model instance | yes, at parse time | `Field(description=...)` |
| `TypedDict` | plain `dict` | none | `Annotated[t, ..., "..."]` |
| `@dataclass` | dataclass instance | none | docstring / comments |

### Structured output from an agent

An agent answers with a transcript, so the schema goes in as `response_format` and the parsed object
comes out on its own key:

```python
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(model="google_genai:gemini-2.5-flash", response_format=ContactInfo)

result = agent.invoke({"messages": [
    {"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}
]})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

`result["messages"]` still holds the raw transcript; `result["structured_response"]` is the parsed
object. An agent with no `tools` and a `response_format` is just an extractor — a perfectly good use
of one.

The same call takes a `TypedDict` (returns a dict) or a `@dataclass` (returns an instance), with the
docstring and `#` comments standing in for `Field(description=...)`:

```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str      # The name of the person
    email: str     # The email address of the person
    phone: str     # The phone number of the person
```

---

## 33. Agent middleware

> Notebook: [middleware.ipynb](Langchainupdated/updatedlangchain/middleware.ipynb)

`create_agent` (§28) hides the loop. **Middleware** is the supported way back in — hooks that run
around each step without rewriting the agent:

- tracking behaviour: logging, analytics, debugging
- transforming prompts, tool selection, output formatting
- retries, fallbacks, early termination
- rate limits, guardrails, PII detection

Middleware is passed as a list, and each entry wraps the whole agent loop:

```python
agent = create_agent(model=..., tools=[...], checkpointer=..., middleware=[...])
```

### The checkpointer is what makes turns connect

Both middlewares below need memory across calls, and in v1 that is a **checkpointer** — the
LangGraph replacement for §25's `RunnableWithMessageHistory`:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model="groq:openai/gpt-oss-120b", checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "test-1"}}
agent.invoke({"messages": [HumanMessage(content="What is 2+2?")]}, config)
```

`thread_id` plays the part `session_id` played in §25: same id, same conversation. `InMemorySaver`
keeps it in the process — swap in a persistent saver and the same code survives a restart.

### `SummarizationMiddleware`

Instead of dropping old turns the way `trim_messages` does (§25), summarization **compresses** them:
when the transcript crosses a threshold, older messages are replaced by a model-written summary and
the most recent ones are kept verbatim.

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:openai/gpt-oss-120b",   # the summariser - can be a cheaper model
            trigger=("messages", 10),           # summarise once the transcript hits 10 messages
            keep=("messages", 4),               # keep the last 4 verbatim
        )
    ],
)
```

Watching `len(response["messages"])` over six questions shows it fire:

```
2  ->  4  ->  6  ->  8  ->  10  ->  6
                                   ^ trigger hit: older turns collapsed into a summary
```

`trigger` and `keep` are `(unit, value)` tuples, and the unit can be counted three ways:

| Unit | `trigger` means | Use when |
| --- | --- | --- |
| `("messages", 10)` | 10 messages in the transcript | turns are roughly uniform in size |
| `("tokens", 550)` | 550 tokens of context | messages vary wildly — tool output especially |
| `("fraction", 0.005)` | 0.5% of the model's context window | the threshold should follow the model, not a constant |

`fraction` is the portable one: the same `0.8` means something sensible on a 8k model and on a 128k
model. The tiny values above (`0.005` / `0.002`) are there to make it trigger on a short test —
production numbers look more like `trigger=0.8, keep=0.3`.

A tool-calling agent hits these limits fast, because tool output lands in the transcript too:

```python
@tool
def search_hotels(city: str) -> str:
    """Search hotels - returns long response to use more tokens."""
    return f"Hotels in {city}: Grand Hotel $350, City Inn $180, Budget Stay $75"

agent = create_agent(
    model="groq:openai/gpt-oss-20b",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model="groq:openai/gpt-oss-20b", trigger=("tokens", 550), keep=("tokens", 200),
    )],
)
```

Across six cities the transcript holds at ~5 messages instead of growing to ~25 — the middleware is
summarising on nearly every turn.

Two things worth being clear about: the hand-rolled `count_tokens` in the notebook
(`len(str(m.content)) // 4`) is a rough character heuristic and is **not** the counter the
middleware uses, so the printed numbers only track the trend; and summarisation costs an extra
model call each time it fires, which is why the trigger should not be set this low for real work.

### `HumanInTheLoopMiddleware`

Some tool calls should not fire unattended — database writes, payments, anything outbound. This
middleware **pauses the graph** before the call and waits for a decision.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="groq:openai/gpt-oss-20b",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),                # required - the pause is persisted state
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,        # reads run unattended
            }
        )
    ],
)
```

Per-tool config is the whole design: reading is harmless, sending is not.

**Step 1 — the agent pauses.** `invoke` returns early, with an `__interrupt__` key alongside the
messages:

```python
result = agent.invoke({"messages": [HumanMessage(content="Send email to john@test.com ...")]}, config)

"__interrupt__" in result
# Interrupt(value={'action_requests': [{'name': 'send_email_tool',
#                                       'args': {...},
#                                       'description': 'Tool execution requires approval...'}],
#                  'review_configs': [{'action_name': 'send_email_tool',
#                                      'allowed_decisions': ['approve', 'edit', 'reject']}]}, ...)
```

The pending call is in `action_requests` — name and args, exactly what is about to run. Nothing has
executed yet.

**Step 2 — resume with a decision.** The second `invoke` sends a `Command` instead of new messages,
on the *same* `thread_id`:

```python
from langgraph.types import Command

result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
result["messages"][-1].content
# '✅ Email sent to john@test.com with subject "Hello" and body "How are you?"'
```

`decisions` is a list because one turn can queue several tool calls. The three types:

| Decision | Effect |
| --- | --- |
| `{"type": "approve"}` | run the tool as the model asked |
| `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}` | run it with corrected arguments |
| `{"type": "reject"}` | refuse; the model is told and continues without the result |

Editing is the interesting one — a human fixes the arguments before anything happens:

```python
result = agent.invoke(
    Command(resume={"decisions": [{
        "type": "edit",
        "edited_action": {
            "name": "send_email_tool",
            "args": {
                "recipient": "correct@email.com",
                "subject": "Corrected Subject",
                "body": "This was edited by human before sending",
            },
        },
    }]}),
    config=config,
)
```

```
invoke -> model -> tool call -> [PAUSE] -> Command(resume=...) -> tool runs -> model -> answer
                                  ^                                    |
                                  +---- pauses again on the next -----+
                                        guarded tool call
```

That last arrow is not hypothetical: in the edit run the agent called `send_email_tool` a second
time (the user's original message still named the wrong address), so the result came back **paused
again**, with `__interrupt__` present and `result["messages"][-1].content` empty — which is why the
printed line was blank. One resume approves one call, not the session.

---

# Part 4 — LangGraph

## 34. Building a graph from scratch

> Notebook: [simplegraph.ipynb](AgenticAIWorkSpace/Langgraph-basics/simplegraph.ipynb)
> · Project: [AgenticAIWorkSpace/](AgenticAIWorkSpace/) — plain `venv` + `requirements.txt`

§28 noted that `create_agent` returns a `CompiledStateGraph` without saying what one *is*. This is
that, from the bottom: a graph is **state**, **nodes** that update it, and **edges** that decide
what runs next.

```bash
cd AgenticAIWorkSpace
python -m venv venv && venv\Scripts\activate     # or: source venv/Scripts/activate
pip install -r requirements.txt                  # langchain, langgraph, langchain-core, langchain-community
```

### State — the schema everything shares

The state is one `TypedDict`, and it is the input schema for every node and edge in the graph:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_info: str
```

### Nodes — plain Python functions

A node takes the state as its first positional argument and returns **the keys it wants to change**,
not the whole state:

```python
def start_play(state: State):
    print("Start_Play node has been called")
    return {"graph_info": state["graph_info"] + " I am planning to play"}

def cricket(state: State):
    return {"graph_info": state["graph_info"] + " Cricket"}

def badminton(state: State):
    return {"graph_info": state["graph_info"] + " Badminton"}
```

There is no LLM here on purpose — a node is just a function, and a graph of functions is easier to
reason about than a graph of model calls. By default a returned key **overwrites** the old value;
the `+` above is doing the appending by hand. (Making a key accumulate automatically is a
*reducer*, e.g. `Annotated[list, add_messages]` — which is exactly how the `messages` key in every
agent graph collects a transcript instead of replacing it.)

### Conditional edges — a router that returns a node name

A routing function also takes the state, but returns **the name of the next node** as a string. The
`Literal` return type is what tells LangGraph which targets are possible, so it can draw and
validate them:

```python
import random
from typing import Literal

def random_play(state: State) -> Literal["cricket", "badminton"]:
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"
```

### Construction and compilation

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

graph.add_edge(START, "start_play")                    # where input enters
graph.add_conditional_edges("start_play", random_play) # branch on the router's return
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

graph_builder = graph.compile()
```

`START` and `END` are special nodes: `START` is where the user input is handed in, `END` is a
terminal. `compile()` runs structural checks — unreachable nodes, edges to names that do not exist
— and returns the runnable `CompiledStateGraph`.

```
          START
            |
        start_play
            |
      random_play (conditional)
        /         \
   cricket      badminton
        \         /
           END
```

`compile()` also gives you the picture for free:

```python
from IPython.display import Image, display

display(Image(graph_builder.get_graph().draw_mermaid_png()))
```

### Invocation

```python
graph_builder.invoke({"graph_info": "Hey My name is shaurya"})
# Start_Play node has been called
# My badminton node has been called
# {'graph_info': 'Hey My name is shaurya I am planning to play Badminton'}
```

The input is the initial state, and the return is the **final state** — not a message, not an
answer. Every node that ran folded its update into it on the way through, and the branch taken was
decided at runtime by `random_play`. Swap that coin flip for an LLM deciding which tool to call and
you have rebuilt §28's agent.

---

## 35. Gotchas worth remembering

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

13. **`OllamaEmbeddings()` has no default model.** Constructing it with no arguments raises a
    pydantic `ValidationError: model — Field required`. Always pass
    `OllamaEmbeddings(model="nomic-embed-text")`.

14. **Ollama has to actually be running**, or every embed call dies with
    `ConnectionError: Failed to connect to Ollama`. Two separate causes:
    the server is not started (`ollama serve`, or launch the desktop app), **or** name
    resolution — on Windows `localhost` resolves to IPv6 `::1` first while Ollama listens only on
    IPv4 `127.0.0.1`. Pinning `base_url="http://127.0.0.1:11434"` removes the second one.

15. **`CharacterTextSplitter` does not enforce `chunk_size`.** It splits on a single separator
    (`"\n\n"` by default), so a paragraph longer than the limit stays whole and it just warns:
    `Created a chunk of size 1370, which is longer than the specified 1000`. Use
    `RecursiveCharacterTextSplitter` when the limit actually matters.

16. **`RunnableWithMessageHistory` needs `input_messages_key` on a dict-input chain.** Wrapping a
    bare model works without it; wrapping `prompt | model` does not, and the string must match the
    `MessagesPlaceholder(variable_name=...)`.

17. **LangServe wraps the payload.** POST bodies go in as `{"input": {...}}` and answers come back
    under `"output"` — the chain's own schema sits one level down.

18. **Groq model ids are namespaced.** It is `"openai/gpt-oss-20b"`, not `"gpt-oss-20b"` — the
    prefix is the model's origin, not the provider being called.

19. **`RunnableWithMessageHistory` is deprecated.** It still works, but every construction warns
    `Use LangGraph's built-in persistence instead`. Fine for learning the mechanics; reach for
    LangGraph checkpointers when building something real.

20. **A too-small trim budget silently lobotomises the bot.** `trim_messages(max_tokens=45)` drops
    the turn that introduced the user's name, and the model then answers "I don't know who you
    are" — no error, just amnesia. The budget is the memory span.

21. **A `VectorStore` is not a `Runnable`.** It cannot go into an LCEL chain directly. Convert it
    with `as_retriever()`, or wrap one method in `RunnableLambda(...).bind(k=1)`.

22. **A plain `dict` inside a chain becomes a `RunnableParallel`.** In
    `{"context": retriever, "question": RunnablePassthrough()} | prompt | llm` both values get the
    *same* input and run side by side, and their results become the prompt's variables. It reads
    like a literal, but it executes.

23. **Re-running `Chroma.from_documents` in the same kernel duplicates the corpus.** The in-memory
    collection survives the cell, so the second run appends a second copy — and searches start
    returning the same text twice under different ids. Restart the kernel, or build the store once.

24. **Different embedding models, different score scales.** Chroma's
    `similarity_search_with_score` returns a **distance** like FAISS (an exact match scores ~0,
    not ~1). Never compare scores across stores or across embedding models.

25. **In v1, `res.content` is not always a string.** Gemini returns a list of content blocks —
    `[{'type': 'text', 'text': '...', 'extras': {...}}]` — because one message can carry text,
    reasoning and tool calls at once. Use **`res.text`** when all you want is the answer.

26. **`GOOGLE_API_KEY` wins over `GEMINI_API_KEY`.** With both set, `langchain-google-genai` prints
    `Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.` and quietly uses the
    other one — which is a confusing five minutes if only one of them is valid.

27. **`os.environ["X"] = os.getenv("X")` raises `TypeError` if the `.env` was not loaded.** Assigning
    `None` to an environment variable is not allowed, so a forgotten `load_dotenv()` fails on the
    assignment rather than later at the API call. `load_dotenv(find_dotenv())` searches parent
    directories, which is what a notebook in a subfolder needs.

28. **`create_agent` returns a `CompiledStateGraph`, not a chain.** It is a LangGraph object, so it
    has no `|` composition — it takes a state dict (`{"messages": [...]}`) and returns one.

29. **A tool call comes back with empty `content`.** `AIMessage(content='', tool_calls=[...])` is a
    *request*, not an answer. Code that only prints `content` sees nothing and looks broken.

30. **Pass the whole `tool_call` to `tool.invoke()`, not just its args.** `get_weather.invoke(tool_call)`
    returns a `ToolMessage` stamped with the matching `tool_call_id`; `invoke(tool_call["args"])`
    returns a bare string, and pairing the id by hand is where the loop breaks.

31. **A `@tool` docstring is prompt text.** Name, type hints and docstring *are* the schema the model
    chooses from. A vague docstring is a tool called at the wrong moment.

32. **`TypedDict` cannot take defaults.** `budget: float | None = Field(None, description="...")` in a
    `TypedDict` body is silently ignored — only the annotation reaches the model. Use
    `Annotated[float | None, ..., "Budget in millions USD"]`, or switch to a Pydantic model.

33. **`include_raw=True` changes the return type.** `with_structured_output(Movie)` returns a `Movie`;
    with `include_raw=True` it returns `{"raw": ..., "parsed": ..., "parsing_error": ...}` instead.

34. **v1 moved the import paths.** `langchain.messages`, `langchain.tools`, `langchain.chat_models`
    replace the `langchain_core.*` equivalents, and `create_stuff_documents_chain` /
    `create_retrieval_chain` now live in `langchain-classic`. Mixing v0 and v1 imports in one
    environment is how the confusing errors start — which is why Part 3 has its own project.

35. **Middleware needs a checkpointer and a `thread_id`.** Summarization and human-in-the-loop are
    both state that has to survive between `invoke` calls. No `checkpointer=InMemorySaver()`, or no
    `config={"configurable": {"thread_id": ...}}`, and there is nothing to summarise or resume.

36. **One resume approves one tool call, not the session.** After `Command(resume=...)` the agent
    keeps going and pauses again on the next guarded call — the result comes back with
    `__interrupt__` present and `messages[-1].content` empty, because the last message is a tool
    request, not an answer. Loop on `while "__interrupt__" in result`, don't check it once.

37. **A resumed result has no `__interrupt__` left.** Re-running an `if "__interrupt__" in result:`
    block after a successful resume silently does nothing — which is why the reject cell in
    `middleware.ipynb` printed nothing at all. Start a fresh request before testing the next
    decision type.

38. **Summarisation costs an extra model call every time it fires.** A trigger tuned low enough to
    demo (`("fraction", 0.005)`) would be pathological in production; `trigger=0.8, keep=0.3` is the
    shape of a real setting. And the notebook's `len(str(m.content)) // 4` token counter is a
    character heuristic, not the counter the middleware actually uses.

39. **A LangGraph node returns a partial state, and the key is overwritten.** Returning
    `{"graph_info": "..."}` replaces the old value — appending is only automatic when the key has a
    reducer (`Annotated[list, add_messages]`).

40. **A conditional-edge function returns the next node's *name*, not state.** It is a router:
    `-> Literal["cricket", "badminton"]` is what tells LangGraph the possible targets so it can
    validate and draw them.

41. **`draw_mermaid_png()` calls a remote renderer.** It posts the diagram to the Mermaid.INK API,
    so it needs network access and fails offline — `get_graph().draw_ascii()` or `draw_mermaid()`
    (which returns the diagram source) work locally.

---

## Where to go next

- **Graphs with an LLM in them** — §34's nodes are plain functions; the next step is a node that
  calls a model, a `messages` key with the `add_messages` reducer, and a cycle back to the model
  after a tool runs
- **Multi-tool agents** — several tools, and the model picking between them rather than confirming
  the only one available
- **Porting Part 2 to v1** — the RAG chain of §26 rebuilt as an agent with the retriever as a tool
- **Memory that survives a restart** — a persistent checkpointer (SQLite, Postgres) in place of
  §33's `InMemorySaver`
- **Writing custom middleware** — §33 used the two built-ins; the hooks are open for guardrails,
  retries and logging of your own
- **History + retrieval together** — the chatbot of §25 over the RAG chain of §26, with the
  follow-up question rewritten against the conversation before it hits the retriever
- **Retrieval quality** — the `"mmr"` and `"similarity_score_threshold"` search types from §26 in
  anger, plus metadata filtering, hybrid search and re-ranking
- **Persisting the vector store** — `persist_directory` so the corpus is embedded once, not once
  per kernel restart
- **Evaluation** — LangSmith datasets and evaluators over the pipeline
