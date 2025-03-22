// Copyright (c) Meta Platforms, Inc. and affiliates.

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.
mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use cgroupfs::CgroupReader;
use clap::Parser;
use libbpf_rs::skel::OpenSkel;
use libbpf_rs::skel::Skel;
use libbpf_rs::skel::SkelBuilder;
use libbpf_rs::MapCore as _;
use libbpf_rs::OpenObject;
use log::debug;
use log::info;
use log::trace;
use maplit::hashmap;
use scx_utils::import_enums;
use scx_utils::init_libbpf_logging;
use scx_utils::scx_enums;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPUS_POSSIBLE;
use scx_utils::NR_CPU_IDS;

const MAX_CELLS: usize = bpf_intf::consts_MAX_CELLS as usize;

/// scx_mitosis: A dynamic affinity scheduler
///
/// Cgroups are assigned to a dynamic number of Cells which are assigned to a
/// dynamic set of CPUs. The BPF part does simple vtime scheduling for each cell.
///
/// Userspace makes the dynamic decisions of which Cells should be merged or
/// split and which cpus they should be assigned to.
#[derive(Debug, Parser)]
struct Opts {
    /// Enable verbose output, including libbpf details. Specify multiple
    /// times to increase verbosity.
    #[clap(short = 'v', long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Exit debug dump buffer length. 0 indicates default.
    #[clap(long, default_value = "0")]
    exit_dump_len: u32,

    /// Interval to consider reconfiguring the Cells (e.g. merge or split)
    #[clap(long, default_value = "10")]
    reconfiguration_interval_s: u64,

    /// Interval to consider rebalancing CPUs to Cells
    #[clap(long, default_value = "5")]
    rebalance_cpus_interval_s: u64,

    /// Interval to report monitoring information
    #[clap(long, default_value = "1")]
    monitor_interval_s: u64,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

#[derive(Debug)]
struct Cell {
    cpu_assignment: BTreeSet<u32>,
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    cells: BTreeMap<u32, Cell>,
    cgroup_to_cell: HashMap<String, u32>,
    prev_percpu_cell_cycles: Vec<[u64; MAX_CELLS]>,
    monitor_interval: std::time::Duration,
}

impl<'a> Scheduler<'a> {
    fn init(opts: &Opts, open_object: &'a mut MaybeUninit<OpenObject>) -> Result<Self> {
        let topology = Topology::new()?;

        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.verbose > 1);
        init_libbpf_logging(None);
        let mut skel = scx_ops_open!(skel_builder, open_object, mitosis)?;

        skel.struct_ops.mitosis_mut().exit_dump_len = opts.exit_dump_len;

        skel.maps.rodata_data.slice_ns = scx_enums.SCX_SLICE_DFL;

        if opts.verbose >= 1 {
            skel.maps.rodata_data.debug = true;
        }
        skel.maps.rodata_data.nr_possible_cpus = *NR_CPUS_POSSIBLE as u32;
        for cpu in topology.all_cores.keys() {
            skel.maps.rodata_data.all_cpus[cpu / 8] |= 1 << (cpu % 8);
        }

        let skel = scx_ops_load!(skel, mitosis, uei)?;

        // Initial configuration
        let mut cells: BTreeMap<u32, Cell> = BTreeMap::new();
        let mut num_cells = 1;
        let mut root_cell = Cell {
            cpu_assignment: topology
                .all_cpus
                .keys()
                .cloned()
                .map(|x| x as u32)
                .collect(),
        };
        let mut cgroup_to_cell = hashmap! {
            "".to_string() => 0
        };

        let mut stack = VecDeque::new();
        let root = CgroupReader::root()?;
        for child in root.child_cgroup_iter()? {
            stack.push_back(child);
        }
        while let Some(reader) = stack.pop_back() {
            for child in reader.child_cgroup_iter()? {
                stack.push_back(child);
            }
            let cpuset = match reader.read_cpuset_cpus() {
                Err(cgroupfs::Error::IoError(_, e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    continue;
                }
                r => r,
            }
            .with_context(|| {
                format!(
                    "Error while reading cpuset from {}",
                    reader.name().display()
                )
            })?;
            if !cpuset.cpus.is_empty() {
                trace!(
                    "Cgroup {} has non-empty cpuset: {}",
                    reader.name().display(),
                    cpuset
                );
                let cg_name = reader.name().to_string_lossy().to_string();
                if !cpuset.cpus.is_subset(&root_cell.cpu_assignment) {
                    bail!(
                        "cpuset of cgroup {} ({:?}) is not a subset of the root cell cpus ({:?})",
                        reader.name().display(),
                        cpuset.cpus,
                        root_cell.cpu_assignment
                    );
                }
                root_cell
                    .cpu_assignment
                    .retain(|x| !cpuset.cpus.contains(x));
                cells.insert(
                    num_cells,
                    Cell {
                        cpu_assignment: cpuset.cpus,
                    },
                );
                cgroup_to_cell.insert(cg_name, num_cells);
                num_cells += 1;
            }
        }
        cells.insert(0, root_cell);

        Ok(Self {
            skel,
            cells,
            cgroup_to_cell,
            prev_percpu_cell_cycles: vec![[0; MAX_CELLS]; *NR_CPU_IDS],
            monitor_interval: std::time::Duration::from_secs(opts.monitor_interval_s),
        })
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        self.update_cgroup_to_cell_assignment()?;
        self.assign_cpus()?;
        let _struct_ops = scx_ops_attach!(self.skel, mitosis)?;
        info!("Mitosis Scheduler Attached");
        while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&self.skel, uei) {
            std::thread::sleep(self.monitor_interval);
            self.debug()?;
            if self.skel.maps.bss_data.user_global_seq != self.skel.maps.bss_data.global_seq {
                trace!("BPF reconfiguration still in progress, skipping further changes");
                continue;
            }
        }
        uei_report!(&self.skel, uei)
    }

    fn update_cgroup_to_cell_assignment(&mut self) -> Result<()> {
        for (cgroup, cell_idx) in self.cgroup_to_cell.iter() {
            let mut cg_path = String::from("/sys/fs/cgroup/");
            cg_path.push_str(cgroup);
            let cg_inode = std::fs::metadata(&cg_path)?.ino();
            let cg_inode_slice = unsafe { any_as_u8_slice(&cg_inode) };
            let cell_idx_u32 = *cell_idx as libc::__u32;
            let cell_idx_slice = unsafe { any_as_u8_slice(&cell_idx_u32) };

            self.skel
                .maps
                .cgrp_init_cell_assignment
                .update(
                    cg_inode_slice,
                    cell_idx_slice,
                    libbpf_rs::MapFlags::NO_EXIST,
                )
                .with_context(|| {
                    format!("Failed to update cgroup cell assignment for: {}", cg_path)
                })?;
            trace!(
                "Assigned {} with inode {} to {}",
                cgroup,
                cg_inode,
                cell_idx
            );
        }
        self.skel.maps.bss_data.update_cell_assignment = true;
        Ok(())
    }

    fn assign_cpus(&mut self) -> Result<()> {
        for (cell_idx, cell) in self.cells.iter() {
            for cpu in cell.cpu_assignment.iter() {
                trace!("Assigned CPU {} to Cell {}", cpu, cell_idx);
                self.skel.maps.bss_data.cells[*cell_idx as usize].cpus[(cpu / 8) as usize] |=
                    1 << cpu % 8;
            }
        }
        Ok(())
    }

    /// Output various debugging data like per cell stats, per-cpu stats, etc.
    fn debug(&mut self) -> Result<()> {
        let zero = 0 as libc::__u32;
        let zero_slice = unsafe { any_as_u8_slice(&zero) };
        if let Some(v) = self
            .skel
            .maps
            .cpu_ctxs
            .lookup_percpu(zero_slice, libbpf_rs::MapFlags::ANY)
            .context("Failed to lookup cpu_ctxs map")?
        {
            for (cpu, ctx) in v.iter().enumerate() {
                let cpu_ctx = unsafe {
                    let ptr = ctx.as_slice().as_ptr() as *const bpf_intf::cpu_ctx;
                    &*ptr
                };
                let diff_cycles: Vec<i64> = self.prev_percpu_cell_cycles[cpu]
                    .iter()
                    .zip(cpu_ctx.cell_cycles.iter())
                    .map(|(a, b)| (b - a) as i64)
                    .collect();
                self.prev_percpu_cell_cycles[cpu] = cpu_ctx.cell_cycles;
                trace!("CPU {}: {:?}", cpu, diff_cycles);
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    let llv = match opts.verbose {
        0 => simplelog::LevelFilter::Info,
        1 => simplelog::LevelFilter::Debug,
        _ => simplelog::LevelFilter::Trace,
    };
    let mut lcfg = simplelog::ConfigBuilder::new();
    lcfg.set_time_level(simplelog::LevelFilter::Error)
        .set_location_level(simplelog::LevelFilter::Off)
        .set_target_level(simplelog::LevelFilter::Off)
        .set_thread_level(simplelog::LevelFilter::Off);
    simplelog::TermLogger::init(
        llv,
        lcfg.build(),
        simplelog::TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    )?;

    debug!("opts={:?}", &opts);

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    let mut open_object = MaybeUninit::uninit();
    loop {
        let mut sched = Scheduler::init(&opts, &mut open_object)?;
        if !sched.run(shutdown.clone())?.should_restart() {
            break;
        }
    }

    Ok(())
}
