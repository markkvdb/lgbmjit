//! Code generation settings.

/// How the compiled code is laid out. Results do not depend on these values; only speed does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Trees per compiled function. Each function loops over the rows of a tile.
    pub block_trees: usize,
    /// Rows converted to keys and run through every function before moving on.
    pub tile_rows: usize,
    /// Subtrees with at most this many leaves are computed without branches.
    pub branch_free_leaves: u32,
    /// Code reached by fewer than this fraction of a tree's training rows is placed out of line.
    pub cold_fraction: f64,
}

impl Settings {
    /// Defaults measured on the load forecast model: Apple M5 Pro for aarch64, AMD Zen 1 otherwise.
    pub fn default_for_host() -> Settings {
        if cfg!(target_arch = "aarch64") { Settings { block_trees: 17, tile_rows: 2048, branch_free_leaves: 4, cold_fraction: 0.01 } }
        else { Settings { block_trees: 8, tile_rows: 1024, branch_free_leaves: 0, cold_fraction: 0.01 } }
    }
}
