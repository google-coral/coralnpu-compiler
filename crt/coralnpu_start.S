/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// A starting functions for simple coralnpu programs.

.extern coralnpu_exception_handler
.extern __fini_array_start__
.extern __fini_array_end__

/**
 * Entry point.
 */
        .section ._init
        .balign 4
        .global _start
        .weak _start
        .type _start, @function
_start:
        csrw minstret, 0
        csrw minstreth, 0
        csrr a0, minstret
        csrr a1, minstreth
        ###############################################
        # Put all scalar registers into a known state #
        ###############################################
.option norelax
        la   sp, __stack_end__
        la   gp, _global_pointer
.option relax

        mv   tp, zero
        mv   t1, zero
        mv   t2, zero
        mv   s0, zero
        mv   s1, zero
        mv   a1, zero
        mv   a2, zero
        mv   a3, zero
        mv   a4, zero
        mv   a5, zero
        mv   a6, zero
        mv   a7, zero
        mv   s2, zero
        mv   s3, zero
        mv   s4, zero
        mv   s5, zero
        mv   s6, zero
        mv   s7, zero
        mv   s8, zero
        mv   s9, zero
        mv   s10, zero
        mv   s11, zero
        mv   t3, zero
        mv   t4, zero
        mv   t5, zero
        mv   t6, zero

        # Zero out the bss section
        la   a0, __bss_start__
        la   a1, __bss_end__
        call crt_section_clear

        # Run constructors
        la   s0, __init_array_start__
        la   s1, __init_array_end__
        bgeu s0, s1, init_array_loop_end
init_array_loop:
        lw   t0, 0(s0)
        jalr t0
        addi s0, s0, 0x4
        bltu s0, s1, init_array_loop
init_array_loop_end:

        # Set up a default trap vector, which will
        # simply call ebreak.
        # Users who require real trap handling should
        # install their own trap vector.
        la t0, coralnpu_exception_handler
        csrw mtvec, t0

        # Set FS (0x6000) and VS (0x600) to Dirty
        li      t0, 0x6600
        csrrs   zero, mstatus, t0

        # Set up sentinel value in _ret
        # If we see this after an ebreak,
        # the break source is unlikely to
        # be a clean return from main.
        la   t0, _ret
        li   a0, 0x0badd00d
        sw   a0, 0(t0)

        #############
        # Call main #
        #############
        li   a0, 0  # argv
        li   a1, 0  # argc
        la   ra, main
        jalr ra, ra
        # Save return value
        mv   s2, a0

        # Call C++ finalizers
        call __cxa_finalize

        # Initialize finalization arrays (destructors)
        # These are called in reverse order.
        la   s0, __fini_array_start__
        la   s1, __fini_array_end__
        beq  s0, s1, fini_array_loop_end
fini_array_loop:
        addi s1, s1, -0x4
        lw   t0, 0(s1)
        jalr t0
        bne  s0, s1, fini_array_loop
fini_array_loop_end:

        # Restore the application's return value
        mv   a0, s2
        la   t0, _ret
        sw   a0, 0(t0)
        beqz a0, success
failure:
        ebreak
        j    loop
success:
        # Commit instruction retirement counters to architectural state.
        # This is for co-simulation verification purposes.
        csrr a0, minstret
        csrr a1, minstreth
        .word 0x08000073 # mpause
loop:
        j    loop

