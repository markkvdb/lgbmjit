//! Helpers shared by the integration tests.
#![allow(dead_code)]
use lgbmjit_core::model::Forest;
use std::path::PathBuf;

/// Directory holding the bundled test models.
pub fn models_dir() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/models") }

/// Text of the bundled model `name`.
pub fn model_text(name: &str) -> String { std::fs::read_to_string(models_dir().join(name).join("model.txt")).unwrap() }

/// Parsed bundled model `name`.
pub fn load(name: &str) -> Forest { Forest::parse(&model_text(name)).unwrap() }

/// Small deterministic random generator for tests (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
}

/// Model text with one tree: a single node on feature 0 with leaves 1.0 (left) and 2.0 (right).
pub fn one_node_text(threshold: f64, decision_type: u8, cat_threshold: Option<u32>, objective: &str) -> String {
    let (infos, cat) = match cat_threshold {
        Some(bits) => ("0:1".to_string(), format!("num_cat=1\ncat_boundaries=0 1\ncat_threshold={bits}\n")),
        None => ("[-10:10]".to_string(), "num_cat=0\n".to_string()),
    };
    format!(
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n\
         objective={objective}\nfeature_names=x\nfeature_infos={infos}\ntree_sizes=0\n\n\
         Tree=0\nnum_leaves=2\n{cat}split_feature=0\nsplit_gain=1\nthreshold={threshold:?}\n\
         decision_type={decision_type}\nleft_child=-1\nright_child=-2\nleaf_value=1 2\nleaf_weight=1 1\n\
         leaf_count=1 1\ninternal_value=0\ninternal_weight=0\ninternal_count=2\nis_linear=0\nshrinkage=1\n\n\n\
         end of trees\n\npandas_categorical:null\n"
    )
}

/// Names of all bundled models.
pub fn model_names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(models_dir()).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().join("model.txt").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}
