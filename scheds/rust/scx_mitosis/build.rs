// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

fn main() {
    scx_cargo::BpfBuilder::new()
        .unwrap()
        .enable_intf("src/bpf/intf.h", "bpf_intf.rs")
        .enable_skel("src/bpf/mitosis.bpf.c", "bpf")
        .add_source("../../../lib/atq.bpf.c")
        .add_source("../../../lib/cgroup_bw.bpf.c")
        .add_source("../../../lib/minheap.bpf.c")
        .add_source("../../../lib/sdt_alloc.bpf.c")
        .compile_link_gen()
        .unwrap();
}
