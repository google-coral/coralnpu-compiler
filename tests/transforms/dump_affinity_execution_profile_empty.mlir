// RUN: %coralnpu_compile --coralnpu-dump-affinity-profile-format=pretty %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-PRETTY

// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY: Execution Profile by Affinity:
// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY:   No dispatches
// CHECK-PRETTY: ========================================================================

func.func @main() {
  return
}
