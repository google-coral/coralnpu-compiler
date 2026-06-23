// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>

// Minimal C++ guard implementations for single-threaded bare-metal systems.

namespace __cxxabiv1 {

// The guard variable is a 64-bit integer.
// The first byte is the flag, 0 = not initialized, 1 = initialized.
using __guard = int64_t;

extern "C" {

// Called to acquire the lock on the guard variable.
// *guard_object: Pointer to the guard variable.
// Returns 1 if the object should be initialized, 0 otherwise.
int __cxa_guard_acquire(__guard* guard_object) {
  // If the first byte is 0, initialization is needed.
  if (*(reinterpret_cast<char*>(guard_object)) == 0) {
    return 1;
  }
  return 0;
}

// Called to release the lock on the guard variable after initialization.
// *guard_object: Pointer to the guard variable.
void __cxa_guard_release(__guard* guard_object) {
  // Set the first byte to 1 to indicate initialization is complete.
  *(reinterpret_cast<char*>(guard_object)) = 1;
}

// Called if initialization fails and the guard needs to be aborted.
void __cxa_guard_abort(__guard* guard_object) {
  // In a bare-metal system, there's not much to do here.
  // We can leave it empty or add some debug output if needed.
}

struct atexit_entry {
  void (*destructor)(void*);
  void* arg;
};

#define MAX_ATEXIT_ENTRIES 8
static atexit_entry atexit_entries[MAX_ATEXIT_ENTRIES];
static int atexit_count = 0;

// Called to register a destructor for a global/static object.
int __cxa_atexit(void (*destructor)(void*), void* arg, void* dso_handle) {
  if (atexit_count >= MAX_ATEXIT_ENTRIES) {
    return -1;
  }
  atexit_entries[atexit_count].destructor = destructor;
  atexit_entries[atexit_count].arg = arg;
  atexit_count++;
  return 0;
}

// Called to execute all registered destructors.
void __cxa_finalize(void* dso_handle) {
  for (int i = atexit_count - 1; i >= 0; i--) {
    if (atexit_entries[i].destructor) {
      atexit_entries[i].destructor(atexit_entries[i].arg);
    }
  }
  atexit_count = 0;
}

// Called if a pure virtual function is called.
void __cxa_pure_virtual() { asm volatile("ebreak"); }

}  // extern "C"

}  // namespace __cxxabiv1
