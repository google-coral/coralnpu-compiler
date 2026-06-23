/* Copyright 2025 Google LLC. */
/* Licensed under the Apache License, Version 2.0, see LICENSE for details. */
/* SPDX-License-Identifier: Apache-2.0 */

MEMORY {
    ITCM(rx): ORIGIN = 0x00000000, LENGTH = @@ITCM_LENGTH@@K
    DTCM(rw): ORIGIN = @@DTCM_ORIGIN@@, LENGTH = @@DTCM_LENGTH@@K
    EXTMEM(rw): ORIGIN = 0x20000000, LENGTH = 4096K
    DDR(rw): ORIGIN = 0x80000000, LENGTH = 2048M
}

STACK_SIZE = DEFINED(__stack_size__) ? __stack_size__ : @@STACK_SIZE@@;
__stack_size = STACK_SIZE;
__stack_shift = 7;
__boot_hart = 0;

ENTRY(_start)

SECTIONS {
    /* ITCM data here */
    . = ORIGIN(ITCM);
    .text : ALIGN(16) {
        *(._init)
        *(.text)
        *(.text.*)
        . = ALIGN(16);
    } > ITCM

    .init.array : ALIGN(16) {
      __init_array_start = .;
      __init_array_start__ = .;
      *(.init_array)
      *(.init_array.*)
      . = ALIGN(16);
      __init_array_end = .;
      __init_array_end__ = .;
    } > ITCM

    .fini.array : {
      __fini_array_start = .;
      __fini_array_start__ = .;
      KEEP(*(.fini_array))
      KEEP(*(.fini_array.*))
      __fini_array_end = .;
      __fini_array_end__ = .;
    } > ITCM

    .rodata : ALIGN(16) {
      *(.srodata)
      *(.srodata.*)
      *(.rodata)
      *(.rodata.*)
      *(.data.rel.ro)
      *(.data.rel.ro.*)
      . = ALIGN(16);
    } > ITCM

    /* Static Thread Local Storage template */
    .tdata : {
        PROVIDE_HIDDEN (__tdata_start = .);
        *(.tdata .tdata.*)
        *(.gnu.linkonce.td.*)
        PROVIDE_HIDDEN (__tdata_end = .);
    } > DTCM
    PROVIDE (__tdata_size = SIZEOF (.tdata));

    .tbss (NOLOAD) : {
        PROVIDE_HIDDEN (__tbss_start = .);
        PROVIDE_HIDDEN (__tbss_offset = ABSOLUTE (__tbss_start - __tdata_start));
        *(.tbss .tbss.*)
        *(.gnu.linkonce.tb.*)
        *(.tcommon)
        PROVIDE_HIDDEN (__tbss_end = .);
    } > DTCM
    PROVIDE (__tbss_size = SIZEOF (.tbss));

    .htif : ALIGN(16) {
      KEEP(*(.htif))
    } > DTCM

    .data : ALIGN(16) {
      __data_start__ = .;
      __data_start = .;
      /**
      * This will get loaded into `gp`, and the linker will use that register for
      * accessing data within [-2048,2047] of `__global_pointer$`.
      *
      * This is much cheaper (for small data) than materializing the
      * address and loading from that (which will take one extra instruction).
      */
      _global_pointer = . + 0x800;
      __global_pointer$ = . + 0x800;
      *(.sdata)
      *(.sdata.*)
      *(.data)
      *(.data.*)
      /**
       * Memory location for the return value from main,
       * which could be inspected by another core in the system.
       **/
      . = ALIGN(4);
      _ret = .;
      . += 4;
      . = ALIGN(16);
      __data_end__ = .;
      _edata = .;
    } > DTCM

    .bss : ALIGN(16) {
      __bss_start__ = .;
      __bss_start = .;
      *(.sbss)
      *(.sbss.*)
      *(.bss)
      *(.bss.*)
      __bss_end__ = .;
      __bss_end = .;
    } > DTCM

    .noinit (NOLOAD) : ALIGN(16) {
      KEEP(*(.noinit))
      KEEP(*(.noinit.*))
    } > DTCM

    /* EXTMEM data here */
    . = ORIGIN(EXTMEM);
    .extdata : ALIGN(16) {
      __extdata_start__ = .;
      *(.extdata)
      *(.extdata.*)
      __extdata_end__ = .;
    } > EXTMEM

    .extbss (NOLOAD) : ALIGN(16) {
      __extbss_start__ = .;
      *(.extbss)
      *(.extbss.*)
      __extbss_end__ = .;
    } > EXTMEM

    /* DDR data here */
    .ddr_data : ALIGN(16) {
      __ddr_data_start__ = .;
      *(.ddr_data)
      *(.ddr_data.*)
      *(.cnidoom.wad)
      *(.cnidoom.weights)
      __ddr_data_end__ = .;
    } > DDR

    .ddr_bss (NOLOAD) : ALIGN(16) {
      __ddr_bss_start__ = .;
      *(.ddr_bss)
      *(.ddr_bss.*)
      __ddr_bss_end__ = .;
    } > DDR

    .heap : ALIGN(16) {
      __heap_start__ = .;
      __heap_start = .;
      @@HEAP_SIZE_SPEC@@
      __heap_end__ = .;
      __heap_end = .;
    } > @@HEAP_LOCATION@@

    .stack : ALIGN(16) {
      @@STACK_START_SPEC@@
      __stack_start__ = .;
      __stack_start = .;
      . += STACK_SIZE;
      __stack_end__ = .;
    } > DTCM

    _end = .;
}
