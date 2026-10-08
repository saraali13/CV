# PYTHON BASIC SYNTAX CHEAT SHEET
python -m venv cv_env
#source cv_env/bin/activate
python -m venv cv_env
cv_env\Scripts\activate
pip install opencv-python numpy matplotlib scikit-image scipy PyWavelets
pip install jupyter notebook
python --version
pip --version
# 1. VARIABLES & BASIC DATA TYPES

name = "Sara"              # String
age = 20                   # Integer
gpa = 3.58                 # Float
is_student = True          # Boolean
print(name)
# Type checking
print(type(name))
print(type(age))

# 2. INPUT

name = input("Enter your name: ")
# input() always gives a string
age = int(input("Enter your age: "))
price = float(input("Enter price: "))

# 3. IF / ELIF / ELSE

age = 20
if age >= 18 and age <= 25:
    print("Young adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")
# Comparison operators
# ==   equal
# !=   not equal
# >    greater than
# <    less than
# >=   greater/equal
# <=   less/equal
# Logical operators
# and
# or
# not

# 4. FOR and While LOOP
# Basic for loop
for i in range(5):
    print(i)
# range(start, stop)
for i in range(1, 6):
    print(i)
# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)
# Reverse loop
for i in range(5, 0, -1):
    print(i)
#nested loop
for i in range(3):

    for j in range(3):
        print(i, j)
while i <= 5:
    print(i)
    i += 1

#break stops the loop
#continue if i=2, then Skips 2

# 5. LIST

fruits = ["Apple", "Banana", "Mango"]
print(fruits)
# Accessing
print(fruits[0])
print(fruits[1])
# Negative indexing
print(fruits[-1])
# Changing value
fruits[0] = "Orange"
# Adding
fruits.append("Apple")
# Insert at specific position
fruits.insert(1, "Grapes")
# Remove by value
fruits.remove("Banana")
# Remove by index
fruits.pop(0)
# Remove last item
fruits.pop()
# Length
print(len(fruits))
# Check if item exists
if "Mango" in fruits:
    print("Mango exists")
# Loop through list
for fruit in fruits:
    print(fruit)
# List slicing-> list[start:stop:step]
numbers = [10, 20, 30, 40, 50]
print(numbers[1:4])     # 20, 30, 40  
print(numbers[:3])      # first 3
print(numbers[2:])      # from index 2
print(numbers[::-1])    # reverse
#operations
print(len(numbers))
print(max(numbers))
print(min(numbers))
print(sum(numbers))
numbers.sort()
print(numbers)
numbers.reverse()
print(numbers)
#NESTED LIST
students = [
    ["Sara", 90],
    ["Ali", 85],
    ["Ahmed", 78]
]
print(students[0])
print(students[0][0])
print(students[0][1])

# 6. TUPLE

# Tuple cannot normally be changed after creation
coordinates = (10, 20)
print(coordinates[0])
print(coordinates[1])
for value in coordinates:
    print(value)

# 7. SET

numbers = {1, 2, 3, 3, 4}
print(numbers)
# Duplicate 3 is automatically removed
numbers.add(5)
numbers.remove(2)
print(numbers)

# 8. DICTIONARY

student = {
    "name": "Sara",
    "age": 20,
    "gpa": 3.58
}
# Access value
print(student["name"])
print(student["age"])
# Another safer way
print(student.get("name"))
# Add new key
student["city"] = "Karachi"
# Change value
student["age"] = 21
# Delete
del student["city"]
# Check key
if "name" in student:
    print("Name exists")
# Length
print(len(student))
# Keys
for key in student:
    print(key)
# Values
for value in student.values():
    print(value)
# Key + value
for key, value in student.items():
    print(key, value)
#nested dictionary 
students = {
    "Sara": {
        "age": 20,
        "gpa": 3.58
    },

    "Ali": {
        "age": 21,
        "gpa": 3.20
    }
}
print(students["Sara"]["age"])
print(students["Sara"]["gpa"])

for name, details in students.items(): #name-> Sara and details-. age and gpa
    print(name)
    print(details["age"])
    print(details["gpa"])
#LIST OF DICTIONARIES
students = [
    {"name": "Sara", "age": 20},
    {"name": "Ali", "age": 21}
]
for student in students:
    print(student["name"])
    print(student["age"])

# 9. FUNCTIONS
def greet():
    print("Hello!")
greet()
# Function with parameters
def greet_user(name):
    print("Hello", name)
greet_user("Sara")
# Function with return
def add(a, b):
    return a + b
#DEFAULT PARAMETERS
def greet(name="User"):
    print("Hello", name)
# Decimal formatting
price = 250
print(f"Price = {price:.2f}")

# 10. EXCEPTION HANDLING
try:
    number = int(input("Enter number: "))
    print(number)
except ValueError:
    print("Please enter a valid number.")

# 11. OOP - CLASS & OBJECT
class Student:
    # Constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # Method
    def display(self):
        print("Name:", self.name)
        print("Age:", self.age)
# Creating object
student1 = Student("Sara", 20)
# Calling method
student1.display()
# Accessing attributes
print(student1.name)
print(student1.age)
student1.age = 5500000
#OOP - INHERITANCE
class Animal:

    def speak(self):
        print("Animal makes sound")
        
class Dog(Animal):

    def bark(self):
        print("Dog barks")

dog = Dog()
dog.speak()
dog.bark()

x = 10
a=1
b=2
print(a + b)    # Addition
print(a - b)    # Subtraction
print(a * b)    # Multiplication
print(a / b)    # Division
print(a // b)   # Integer/Floor division
print(a % b)    # Remainder
print(a ** b)   # Power

#BOOLEAN OPERATORS
print(x > 5)
print(x < 5)
print(x == 10)
print(x != 10)
print(x > 5 and x < 20)
print(x > 20 or x == 10)
print(not x > 20)

#QUICK OOP STRUCTURE TO MEMORIZE
class ClassName:

    def __init__(self, value):
        self.value = value

    def method(self):
        print(self.value)
object1 = ClassName(10)
object1.method()
