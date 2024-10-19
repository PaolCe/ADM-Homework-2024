# PROBLEM 1

# INTRODUCTION
# Introduction - Say "Hello, World!" With Python

print("Hello, World!")
# -------------------------------

# Introduction - Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 != 0:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
elif n % 2 == 0 and n > 20:
    print("Not Weird")
# -------------------------------

# Introduction - Arithmetic Operators

if __name__ == '__main__':
    a, b = int(input()), int(input())
print(a + b)
print(a - b)
print(a * b)
# -------------------------------

# Introduction - Python: Division

a, b = int(input()), int(input())
print(a // b)
print(a / b)
# -------------------------------

# Introduction - Loops

n = int(input())
for i in range(n):
    print(i ** 2)
# -------------------------------   

# Introduction - Write a function

def is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
year = int(input())
print(is_leap(year))
# -------------------------------   

# Introduction - Print function

n = int(input())
print(*range(1, n + 1), sep="")

# -------------------------------   

# DATA TYPES
# Data types - List Comprehensions

X = int(input())
Y = int(input())
Z = int(input())
N = int(input())

RESULT = []

for a in range(0, X + 1):
    for b in range(0, Y + 1):
        for c in range(0, Z + 1):
            if (a + b + c) != N:
                RESULT.append([a, b, c])

print(RESULT)

# -------------------------------   

# Data types - Find the Runner-Up Score!

n = int(input())
arr = list(map(int, input().split()))
print(max([x for x in arr if x != max(arr)]))

# -------------------------------   

# Data types - Nested Lists

if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])

    students = sorted(students, key = lambda x: x[1])
    second_lowest_score = sorted(list(set([x[1] for x in students])))[1]
    desired_students = []
    for stu in students:
        if stu[1] == second_lowest_score:
            desired_students.append(stu[0])
    print("\n".join(sorted(desired_students)))

 # -------------------------------  
 
 # Data types - Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    total_marks = 0
    for i in student_marks[query_name]:
        total_marks += i
    avg = (total_marks / len(student_marks[query_name]))
    print("{:.2f}".format(avg))

 # -------------------------------  

 # Data types - Lists

if __name__ == '__main__':

    N = int(input())

    List=[];

    for i in range(N):

        command=input().split();

        if command[0] == "insert":

            List.insert(int(command[1]),int(command[2]))

        elif command[0] == "append":

            List.append(int(command[1]))

        elif command[0] == "pop":

            List.pop();

        elif command[0] == "print":

            print(List)

        elif command[0] == "remove":

            List.remove(int(command[1]))

        elif command[0] == "sort":

            List.sort();

        else:

            List.reverse();
 # -------------------------------  

# Data types - Tuples

n = int(input())
t = tuple(map(int, input().split()))
print(hash(t))
 # -------------------------------  

# STRINGS
# Strings - sWAP cASE

def swap_case(s):
    
    result = ""
    for let in s:
        if let.isupper():
            result += let.lower()
        else:
            result += let.upper()
    return result
 # -------------------------------  

 # Strings - String Split and Join

def split_and_join(line):

    a = line.split(" ")

    b = "-".join(a)

    return(b)


if __name__ == '__main__':

    line = input()

    result = split_and_join(line)

    print(result)
 # -------------------------------  

 # Strings - What's Your Name?

 def print_full_name(first, last):
    print("Hello " + first + " " + last + "! You just delved into python.")
 # ------------------------------- 

 # Strings - Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[(position + 1):]
 # ------------------------------- 

 # Strings - Find a string

  M = int(1e9 + 7)
BASE = 128

def count_substring(s, p):
    n = len(p)
    shift = pow(BASE, n, M)
    
    pHash = 0
    for c in p:
        pHash = (pHash * BASE + ord(c)) % M
    
    subHash = cnt = 0
    for i, c in enumerate(s):
        subHash = (subHash * BASE + ord(c)) % M
        if i >= n:
            subHash = (subHash - ord(s[i - n]) * shift) % M
        if subHash == pHash:
            cnt += 1
    return cnt

 # ------------------------------- 
   
 # Strings - String Validators

   if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s) )
    print(any(i.isalpha() for i in s) )
    print(any(i.isdigit() for i in s) )
    print(any(i.islower() for i in s) )
    print(any(i.isupper() for i in s) )
 # -------------------------------

 # Strings - Text Alignment

thickness = int(input())  # This must be an odd number
c = 'H'

# Top cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

# Top pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Middle belt
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

# Bottom pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Bottom cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))
 # ------------------------------- 

  # Strings - Text Wrap

  import textwrap

def wrap(string, max_width):
     paragraph = ''
     count = 0
     for i in range(len(string)):
        if i % max_width == 0 and i != 0:
            paragraph += '\n'
        paragraph += string[i]
     return paragraph
 # ------------------------------- 

 # Strings - Designer Door Mat

n, m = map(int, input().split())

 # Top pattern
for i in range(n // 2):
    pattern = ".|." * (2 * i + 1)
    print(pattern.center(m, "-"))

 # Middle line
print("WELCOME".center(m, "-"))

# Bottom pattern
for i in range(n // 2 - 1, -1, -1):
    pattern = ".|." * (2 * i + 1)
    print(pattern.center(m, "-"))
# ------------------------------- 

# Strings - String Formatting

def print_formatted(number):
    for i in range(number):
        justify_char_length = len(bin(number)[2:])
        first = i + 1
        first_str = str(first).rjust(justify_char_length, ' ')     # decimal
        second = oct(first)[2:].rjust(justify_char_length, ' ')    # octal
        third = hex(first)[2:].rjust(justify_char_length, ' ')
        third_str = str(third).upper()         # hexidecimal
        fourth = bin(first)[2:].rjust(justify_char_length, ' ')    # binary
        print(f"{first_str} {second} {third_str} {fourth}")
# ------------------------------- 

# Strings - String Formatting

def print_rangoli(n):
    a=[chr(i) for i in range(97,123)]
    t=1
    l=[]
    ch=a[n-t]
    for i in range(1,n):
        print("-".join(ch).center(1+4*(n-1),"-"))
        l.append("-".join(ch).center(1+4*(n-1),"-"))
        ch=ch[:i]+a[n-t-1]+ch[-i:]
        t+=1
    print("-".join(ch).center(1+4*(n-1),"-"))
    for i in l[::-1]:
        print(i)
# ------------------------------- 

# Strings - Capitalize!

def solve(s):
    ans = s.split(' ')
    ans1 = (((i.capitalize() for i in ans)))
    return ' '.join(ans1)
# -------------------------------

# Strings - The Minion Game

def minion_game(string):
    # your code goes here
    vowels = 'AEIOU'
    kevin_score = 0
    stuart_score = 0
    length = len(string)

    for i in range(length):
        if string[i] in vowels:
            kevin_score += length - i
        else:
            stuart_score += length - i

    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")
# -------------------------------

# Strings - Merge the Tools!

from collections import OrderedDict as od

def merge_the_tools(string, k):
    l = len(string)//k
    for i in range(l):
        print(''.join(od.fromkeys(string[i*k:(i*k)+k])))
# -------------------------------

# SETS
# Sets - Introduction to Sets

def average(array):
    set1=set()
    total=count=0
    for i in arr:
        if i not in set1:
            set1.add(i)
            total=total+i
            count=count+1
    return total/count
# -------------------------------

# Sets - Symmetric Difference

M = int(input())
a = set(map(int, input().split()))
N = int(input())
b = set(map(int, input().split()))
symmetricDifference = sorted(a ^ b)
for x in symmetricDifference:
    print(x)
# -------------------------------

# Sets - No Idea!

def no_idea(arr, A, B):
    happiness = 0
    for i in arr:
        if i in A:
            happiness += 1
        if i in B:
            happiness -= 1
        else:
            pass
    return happiness


if __name__ == "__main__":
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    A = set(map(int, input().split()))
    B = set(map(int, input().split()))
    print(no_idea(arr, A, B))

# -------------------------------

# Sets - Set .add()

N = int(input())
countries = set()
[countries.add(input()) for _ in range(N)]
print(len(countries))
# -------------------------------

# Sets - Set .discard(), .remove() & .pop()

input()
s  = list(map(int, input().split()))
s.reverse()
s = set(s)
num_commands = int(input())
command = ["", ""]
for i in range(num_commands):
    command = input().split()
    if len(command) == 1:
        getattr(s, command[0])()
    elif len(command) == 2:
        command[1] = int(command[1])
        if command[1] in s:
            getattr(s, command[0])(command[1])
print(sum(s))
# -------------------------------

# Sets - Set .union() Operation

_ = int(input())
SET_N = set(map(int, input().split()))

_ = int(input())
SET_B = set(map(int, input().split()))

NEW_SET = SET_N.union(SET_B)
print(len(NEW_SET))
# -------------------------------

# Sets - Set .intersection() Operation

int(input())
english_subscriptions = set(map(int, input().split()))

int(input())
french_subscriptions = set(map(int, input().split()))

all_subscriptions = english_subscriptions.intersection(french_subscriptions)
print(len(all_subscriptions))
# -------------------------------

# Sets - Set .difference() Operation

e = int(input())

e_rollno = set(map(int, input().split()[:e]))

f = int(input())

f_rollno = set(map(int, input().split()[:f]))

result = e_rollno.difference(f_rollno)

print(len(result))
# -------------------------------

# Sets - Set .symmetric_difference() Operation

a = int(input())
eng = set(map(int, input().split()))
b = int(input())
french = set(map(int, input().split()))

print(len(eng ^ french))
# -------------------------------

# Sets - Set Mutations

A = int(input())
SET_A = set(map(int, input().split()))
N = int(input())

for _ in range(N):
    operation = input().split()
    new_set = set(map(int, input().split()))
    eval('SET_A.{}({})'.format(operation[0], new_set))

print(sum(SET_A))
# -------------------------------

# Sets - The Captain's Room 

K = int(input())
room_list = list(map(int, input().split()))
sorted_room_list = sorted(room_list)
i = 0
while i < len(sorted_room_list):
    if i+1 == len(sorted_room_list):
        print(sorted_room_list[i])
        break
    if sorted_room_list[i] == sorted_room_list[i+1]:
        i += K
    else:
        print(sorted_room_list[i])
        break
# -------------------------------

# Sets - The Captain's Room 

for i in range(int(input())):
    a = int(input())
    set_a = set(map(int, input().split()))

    b = int(input())
    set_b = set(map(int, input().split()))

    if len(set_a - set_b) == 0:
        print("True")
    else:
        print("False")
# -------------------------------

# Sets - The Captain's Room  

A = set(map(int, input().split()))
for _ in range(int(input())):
    X = set(map(int, input().split()))
    if A.issuperset(X) != True or len(A) == len(X): 
        print(False)
        break 
else: print(True)     
# -------------------------------

# COLLECTIONS
#Collections - collections.Counter()

from collections import Counter
X = int(input())
shoe_sizes = Counter(map(int, input().split()))
N = int(input())
total = 0

for _ in range(N):
    size, price = map(int, input().split())
    if shoe_sizes.get(size, 0) > 0:
        shoe_sizes[size] -= 1
        total += price
        if shoe_sizes[size] == 0:
            del shoe_sizes[size]
print(total)
# -------------------------------

#Collections - DefaultDict Tutorial

from collections import defaultdict

n, m = map(int, input().split())
d = defaultdict(list)

for i in range(1, n + 1):
    d[input()].append(i)

for _ in range(m):
    print(*d.get(input(), [-1]))
# -------------------------------

#Collections - Collections.namedtuple()

from collections import namedtuple

n = int(input())
fields = input().split()
students = namedtuple('Student', fields)
total_marks = sum(int(students(*input().split()).MARKS) for _ in range(n))
print(total_marks / n)
# -------------------------------

#Collections - Collections.OrderedDict()

from collections import OrderedDict

a = OrderedDict()
for _ in range(int(input())):
    item, price = input().rsplit(' ', 1)
    a[item] = a.get(item, 0) + int(price)

for item, price in a.items():
    print(item, price)
# -------------------------------

#Collections - Word Order

from collections import Counter

res = Counter(input().strip() for _ in range(int(input())))
print(len(res))
print(*res.values())
# -------------------------------

#Collections - Collections.deque()

from collections import deque

d = deque()
for _ in range(int(input())):
    args = input().split()
    getattr(d, args[0])(*args[1:])

print(' '.join(d))
# -------------------------------

#Collections - Piling Up!

ANS = []
T = int(input())

for _ in range(T):
    n = int(input())
    sl = list(map(int, input().split()))
    a = 0  # Initialize a variable to store the last removed element

    while len(sl) > 1:
        if sl[0] >= sl[-1]:
            a = sl.pop(0)  # Remove and store the front element
        else:
            a = sl.pop()  # Remove and store the last element
        
        if sl and (sl[0] > a or sl[-1] > a):  # Check conditions
            ANS.append("No")
            break
    else:
        ANS.append("Yes")  # Only append "Yes" if the loop completes without break

print("\n".join(ANS))
# -------------------------------

#Collections - Company Logo

import collections

s = sorted(input().strip())
s_counter = collections.Counter(s).most_common()

s_counter = sorted(s_counter, key=lambda x: (x[1] * -1, x[0]))
for i in range(0, 3):
    print(s_counter[i][0], s_counter[i][1])
# -------------------------------

#DATA AND TIME
#Data and Time - Calendar Module

import calendar
month, day, year = list(map(int,input().split()))
ans = calendar.weekday(year, month, day)
print((calendar.day_name[ans]).upper())
# -------------------------------

#Data and Time - Time Delta

from datetime import datetime
import os

def time_delta(t1, t2):
    format_ = '%a %d %b %Y %H:%M:%S %z'
    return str(int(abs((datetime.strptime(t1, format_) - datetime.strptime(t2, format_)).total_seconds())))

if __name__ == '__main__':
    with open(os.environ['OUTPUT_PATH'], 'w') as fptr:
        for _ in range(int(input())):
            t1, t2 = input(), input()
            fptr.write(time_delta(t1, t2) + '\n')
# -------------------------------

#EXCEPTIONS
#Exceptions - Exceptions

for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a // b)
    except Exception as e:
        print("Error Code:", e)
# -------------------------------

#BUILT-INS
#Built-ins - Zipped!

n, s =(input()).split()
res=[]
for _ in range(int(s)):
    x=list(map(float, input().split()))
    res.append(x)
tup=(list(zip(*res)))
for i in tup:
    print(sum(i)/float(s))
# -------------------------------

#Built-ins - Athlete Sort

import sys

if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    arr = []
    for arr_i in range(n):
       arr_t = [int(arr_temp) for arr_temp in input().strip().split(' ')]
       arr.append(arr_t)
    k = int(input().strip())
    
    sorted_arr = sorted(arr, key = lambda x : x[k])
    for row in sorted_arr:
        print(' '.join(str(y) for y in row))
# -------------------------------

#Built-ins - ginortS

import re
string=input()
lc=re.findall(r'[a-z]',string)
uc=re.findall(r'[A-Z]',string)
num=re.findall(r'[0-9]',string)
od=[ i for i in num if int(i)%2!=0]
ev=[ i for i in num if int(i)%2==0]
print(*sorted(lc),*sorted(uc),*sorted(od),*sorted(ev),sep="")
# -------------------------------

#PYTON FUNCTIONALS
#Python Functionals - Map and Lambda Function

cube = lambda x: x ** 3

def fibonacci(n):
    fib_list = [0, 1]
    for i in range(2, n):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list[:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
# -------------------------------

#REGEX AND PARSING
#Regex and Parsing - Detect Floating Point Number

import re

class Main():
    def __init__(self):
        self.n = int(input())
        
        for i in range(self.n):
            self.s = input()
            print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', self.s)))
            
            
if __name__ == '__main__':
    obj = Main()
# -------------------------------

#Regex and Parsing - Re.split()

import re

pattern = r"\D+"

input_string = input()
split_result = re.split(pattern, input_string)

print("\n".join(split_result))
# -------------------------------

#Regex and Parsing - Re.split()

import re

if __name__=="__main__":
    s = input()
    r = re.search(r'([0-9a-zA-Z]{1})\1+',s)
    if r:
        print(r.group(0)[0])
    else:
        print(-1)
 # -------------------------------
 
 #Regex and Parsing - Re.findall() & Re.finditer()

 import re

if __name__=="__main__":
    s = input()
    r = re.findall(r'(?<=[^aeiouAEIOU])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU])',s)
    if not r:
        print(-1)
    else:
        for p in r:
            print(p)
 # -------------------------------

 #Regex and Parsing - Re.start() & Re.end()

 import re

string = input()
substring = input()

pattern = re.compile(substring)
match = pattern.search(string)
if not match: print('(-1, -1)')
while match:
    print('({0}, {1})'.format(match.start(), match.end() - 1))
    match = pattern.search(string, match.start() + 1)
 # -------------------------------

 #Regex and Parsing - Regex Substitution

 import re

def substitute_text_in_string(text: str):
    """
    Substitute text in string
    && -> and
    || -> or
    change only if there are spaces around
    """
    pattern = r'(?<=\s)&&(?=\s)|(?<=\s)\|\|(?=\s)'
    output = re.sub(pattern, lambda x: 'and' if x.group() == '&&' else 'or', text)
    print(output)
    

if __name__ == '__main__':
    n = int(input())
    text = ''
    for _ in range(n):
        text += input() + '\n'

    substitute_text_in_string(text)
 # -------------------------------

 #Regex and Parsing - Validating Roman Numerals

 regex_pattern = r'M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$'
 # -------------------------------

#Regex and Parsing - Validating phone numbers
import re

N = int(input())
for i in range(0, N):
    # Must start with a 7, 8, or 9 according to the problem, must only contain 10 digits total
    print('YES') if re.match(r'[789]\d{9}$', input()) else print('NO') 
# -------------------------------

#Regex and Parsing - Validating and Parsing Email Addresses

import re
import email.utils 

N = int(input())

pattern = r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}$'
for i in range(0, N):
    parsed_addr = email.utils.parseaddr(input())
    if re.search(pattern, parsed_addr[1]):
        print(email.utils.formataddr(parsed_addr)) 
# -------------------------------

#Regex and Parsing - Hex Color Code

import re
pattern = r'#[\da-fA-F]{3,6}'
res = []
for i in range(int(input())):
    lines = input()
    if len(lines)==0:
        continue
    if lines[0] != '#':
        m = re.findall(pattern, lines)
        [res.append(j) for j in m]
print(*res, sep='\n')
# -------------------------------

#Regex and Parsing - HTML Parser - Part 1

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f'Start : {tag}')
        for att in attrs:
            print(f'-> {att[0]} > {att[1]}')
        
    def handle_endtag(self, tag):
        print(f'End   : {tag}')
        
    def handle_startendtag(self, tag, attrs):
        print(f'Empty : {tag}')
        for att in attrs:
            print(f'-> {att[0]} > {att[1]}')


if __name__ == '__main__':
    n = int(input().rstrip())
    html_text = ''
    for i in range(n):
        html_text += input()
    praser = MyHTMLParser()
    praser.feed(html_text)
# -------------------------------

#Regex and Parsing - HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        print('>>> Multi-line Comment') if (comment.find('\n') != -1) else print('>>> Single-line Comment')
        print(comment)
        
    def handle_data(self, data):
        if data is '\n':
            return
        print('>>> Data')
        print(data)
            
  
  
  
  
  
  
  
  
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
# -------------------------------

#Regex and Parsing - Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

# Same approach as the HTML Parser 1 Challenge! Make a subclass and override methods
N = int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attributes):
        print(tag)
        [print('-> {} > {}'.format(*attribute)) for attribute in attributes]
    
html = '\n'.join([input() for x in range(0, N)])  
parser = MyHTMLParser()
parser.feed(html)
# -------------------------------

#Regex and Parsing - Validating UID

import re
for _ in range(int(input())):
    s = input()
    m = re.search(r'^(?=(?:.*[A-Z]){2})(?=(?:.*\d){3})[A-Za-z0-9]{10}$', s)
    if m and len(s)==len(set(s)):
        print("Valid")
    else:
        print("Invalid")
# -------------------------------

#Regex and Parsing - Validating Credit Card N

import re

pattern_start = r'^'
pattern_no_repetition = r'(?!.*(\d)(-?\1){3})'
credit_card_pattern = r'[456]((\d){15}|(\d){3}(-[\d]{4}){3})'
pattern_end = r'$'

pattern = re.compile(
    pattern_start
    + pattern_no_repetition
    + credit_card_pattern
    + pattern_end
)

for _ in range(int(input())):
    credit_card = input()
    print("Valid" if pattern.search(credit_card) else 'Invalid')
# -------------------------------

#Regex and Parsing -Validating Postal Codes

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)
# -------------------------------

#Regex and Parsing - Matrix Script

import re

matrix = list()
for _ in range(int(input().split()[0])):
    matrix.append(list(input()))

matrix = list(zip(*matrix))

sample = str()
for subset in matrix:
    for letter in subset:
        sample += letter


print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', sample))
# -------------------------------

#XML
#Xml - Find the scores

def get_attr_number(node):
    total = 0
    for child in node.iter():
        total += int(len(child.attrib))
    return total
# -------------------------------

#Xml - XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree
maxdepth = -1
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
        
    for child in elem:
        depth(child, level + 1)
# -------------------------------

#CLOSURE AND DECORATORS
#Closures and Decorators - Standardize Mobile Number Using Decorators


def wrapper(f):
    def fun(l):
        # complete the function
        f(["+91 " + c[-10:-5] + " " + c[-5:] for c in l])
    return fun
 # -------------------------------
 
 #Closures and Decorators - Decorators 2 - Name Directory

 def person_lister(f):
    def inner(people):
        # Sort people by age (index 2 of each sublist, converted to int)
        sp = sorted(people, key=lambda x: int(x[2]))
        
        # Function to determine the title based on gender
        def title(r):
            return "Mr." if r[3] == "M" else "Ms."
        
        # Return the formatted string using the title and names
        return [f"{title(r)} {r[0]} {r[1]}" for r in sp]
    return inner
  # -------------------------------

  #NUMPY
  #Numpy - Arrays

  import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    return(numpy.array(arr[::-1], float))                       
arr = input().strip().split(' ')
result = arrays(arr)
print(result)
  # -------------------------------

  #Numpy - Shape and Reshape

 import numpy as np

array = np.array(list(map(int, input().split())))
array.shape = (3, 3)
print(array)
  # -------------------------------

 #Numpy - Transpose and Flatten

 import numpy

N, M = map(int, input().split())
data = []
for _ in range(N):
    data.append(list(map(int, input().split())))

data = numpy.array(data)
print(data.transpose())
print(data.flatten())
  # -------------------------------

 #Numpy - Concatenate

 import numpy

N, M, P = map(int, input().split())

data1 = []
for _ in range(N):
    data1.append(list(map(int, input().split())))

data2 = []
for _ in range(M):
    data2.append(list(map(int, input().split())))

result = numpy.concatenate((data1, data2), axis=0)
print(result)
  # -------------------------------

 #Numpy - Zeros and Ones

 import numpy

N = tuple(map(int, input().split()))
print(numpy.zeros(N, int))
print(numpy.ones(N, int))
  # -------------------------------

 #Numpy - Eye and Identity

 import numpy

M, N = map(int, input().split())

print(str(numpy.eye(M, N)).replace('1',' 1').replace('0',' 0'))
  # -------------------------------

 #Numpy - Array Mathematics

 import numpy

n,m = input().split()

a1 = numpy.array([list(map(int, input().split())) for i in range(int(n))])
a2 = numpy.array([list(map(int, input().split())) for i in range(int(n))])

print(numpy.add(a1,a2))
print(numpy.subtract(a1,a2))
print(numpy.multiply(a1,a2))
print(numpy.floor_divide(a1,a2))
print(numpy.mod(a1,a2))
print(numpy.power(a1,a2))
 # -------------------------------

 #Numpy - Floor, Ceil and Rint

 import numpy

numpy.set_printoptions(sign=' ')

array = numpy.array(list(map(float, input().split())), dtype=float)
print(numpy.floor(array))
print(numpy.ceil(array))
print(numpy.rint(array))
 # -------------------------------

#Numpy - Sum and Prod

import numpy

N, M = map(int, input().split())
data = []
for _ in range(N):
    data.append(list(map(int, input().split())))

data = numpy.array(data)
data = data.sum(axis=0)
product = data.prod()
print(product)
 # -------------------------------

#Numpy - Min and Max

import numpy

N, M = map(int, input().split())
data = []
for _ in range(N):
    data.append(list(map(int, input().split())))

data = numpy.array(data)
data = data.min(axis=1)
result = data.max()
print(result)
 # -------------------------------

 #Numpy - Mean, Var, and Std

 import numpy

N, M = map(int, input().split())
A = numpy.array([list(map(int, input().split())) for n in range(N)])

print(numpy.mean(A, axis = 1))
print(numpy.var(A, axis = 0))
print(numpy.round(numpy.std(A), 11))
 # -------------------------------

 #Numpy - Dot and Cross

 import numpy
numpy.set_printoptions(legacy='1.13')


def zero(size):
    return [0 for _ in range(size)]


def get_matrix(size):
    matrix = []
    for _ in range(size):
        matrix.append(list(map(int, input().split())))
    return matrix


N = int(input())
matrix1 = numpy.array(get_matrix(N))
matrix2 = numpy.array(get_matrix(N)).transpose()

result = []
for row in range(N):
    result.append(zero(N))
    for column in range(N):
        result[row][column] = int(numpy.dot(matrix1[row], matrix2[column]))

print(numpy.array(result))
 # -------------------------------
 
 #Numpy - Inner and Outer

import numpy
numpy.set_printoptions(legacy='1.13')

A = numpy.array(list(map(int, input().split())))
B = numpy.array(list(map(int, input().split())))

print(numpy.inner(A, B))
print(numpy.outer(A, B))
 # -------------------------------

  #Numpy - Polynomials

  import numpy
numpy.set_printoptions(legacy='1.13')

polynomial = list(map(float, input().split()))
x = float(input())
value = numpy.polyval(polynomial, x)
print(value)
# -------------------------------

#Numpy -Linear Algebra

import numpy

N = int(input())
matrix = []
for _ in range(N):
    matrix.append(list(map(float, input().split())))

result = round(numpy.linalg.det(matrix), 2)
print(result)
# -------------------------------

# PROBLEM 2

# BIRTHDAY CAKE CANDLES
# Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

    count=0
    big = max(ar)
    for i in range(len(ar)):
        if(ar[i]==big):
            count+=1
    return count
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
# -------------------------------

#NUMBER LINE JUMPS
#Number Line Jumps

def kangaroo(x1, v1, x2, v2):
    if x1 < x2 and v1 < v2:
        return 'NO'
    else:
        if v1!=v2 and (x2-x1)%(v2-v1)==0:
            return 'YES' 
        else:
            return 'NO'
# -------------------------------

#VIRAL ADVERTISING
#Viral Advertising

def viralAdvertising(n):
    numOfPeopleAdvertised=5
    
    totNumOfPeopleLiked = 0
    for day in range(n):
        numOfPeopleLiked = numOfPeopleAdvertised//2
        totNumOfPeopleLiked += numOfPeopleLiked
        numOfPeopleAdvertised = numOfPeopleLiked*3
        
        
    return totNumOfPeopleLiked
# -------------------------------

#RECURSIVE DIGITAL SUM
#Recursive Digit Sum

def superDigit(n, k):
    # Write your code here
    if len(n)<2:
        return n
    else:
        Sum=k*sum([int(char) for char in n])
        return superDigit(str(Sum), 1)
# -------------------------------

#INSERT SORT - PART 1
#Insertion Sort - Part 1

def insertionSort1(n, arr):
    tmp = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] > tmp:
            arr[i+1] = arr[i]
            print(' '.join(map(str, arr)))
        else:
            arr[i+1] = tmp
            print(' '.join(map(str, arr)))
            return

    arr[0] = tmp
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
 # -------------------------------

#INSERT SORT - PART 2
#Insertion Sort - Part 2

import math
import os
import random
import re
import sys


def insertionSort2(n, arr):

    for i in range(1, n):
        if arr[i] < arr[i-1]:
            if i == n-1:
                arr = sorted(arr)
                print(*arr)
            else:
                arr[:i+1] = sorted(arr[:i+1])
                print(*arr)
        else:
            print(*arr)

if _name_ == '_main_':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
  # ------------------------------- 
