import os
import sys

print("=== CWD ===")
print(os.getcwd())
print("===========")

print("=== LIST CWD DIR ===")
try:
  print(os.listdir("."))
except Exception as e:
  print("FAILED LIST CWD DIR:", e)
print("====================")

print("=== SYS PATH ===")
for p in sys.path:
  print(p)
print("================")

try:
  import iree.runtime

  print("SUCCESS runtime import")
except Exception as e:
  print("FAILED runtime import:", e)

try:
  import iree.compiler

  print("SUCCESS compiler import")
except Exception as e:
  print("FAILED compiler import:", e)
