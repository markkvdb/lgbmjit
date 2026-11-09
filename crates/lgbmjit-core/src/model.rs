//! Parsing of LightGBM's text model format.
use std::collections::HashMap;
use std::fmt;

/// Why a model could not be loaded.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The text is not a valid LightGBM model.
    Parse(String),
    /// The model uses a feature lgbmjit does not support.
    Unsupported(String),
    /// Cranelift could not compile the generated code.
    Compile(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(m) => write!(f, "invalid LightGBM model: {m}"), Error::Unsupported(m) => write!(f, "unsupported LightGBM model: {m}"),
            Error::Compile(m) => write!(f, "compiling the model failed: {m}")
        }
    }
}

impl std::error::Error for Error {}

fn parse_err(m: impl Into<String>) -> Error { Error::Parse(m.into()) }

/// How a numerical node treats missing values (bits 2 and 3 of the decision type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingType { None, Zero, NaN }

/// The output transform of a model's objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Objective {
    Identity,
    /// `Sign(x) * x * x`, for regression objectives trained with `reg_sqrt`.
    Sqrt,
    Exp,
    /// `1 / (1 + exp(-s * x))`.
    Sigmoid(f64),
    Softmax,
    /// `1 / (1 + exp(-s * x))` for each class.
    OvaSigmoid(f64),
    /// `1 / (1 + exp(-x))`.
    CrossEntropy,
    /// `log1p(exp(x))`.
    CrossEntropyLambda,
}

impl Objective {
    /// Parses the value of a model file's `objective=` line, such as `binary sigmoid:1`.
    pub fn parse(s: &str) -> Result<Objective, Error> {
        let mut tokens = s.split_whitespace();
        let name = tokens.next().unwrap_or("");
        let rest: Vec<&str> = tokens.collect();
        let sigmoid = || -> Result<f64, Error> {
            let v = rest.iter().find_map(|t| t.strip_prefix("sigmoid:")).ok_or_else(|| parse_err(format!("objective `{s}` has no sigmoid parameter")))?;
            v.parse().map_err(|_| parse_err(format!("bad sigmoid parameter in `{s}`")))
        };
        Ok(match name {
            "regression" | "regression_l1" | "huber" | "fair" | "quantile" | "mape" => {
                if rest.contains(&"sqrt") { Objective::Sqrt } else { Objective::Identity }
            }
            "poisson" | "gamma" | "tweedie" => Objective::Exp,
            "binary" => Objective::Sigmoid(sigmoid()?),
            "multiclass" => Objective::Softmax,
            "multiclassova" => Objective::OvaSigmoid(sigmoid()?),
            "cross_entropy" => Objective::CrossEntropy,
            "cross_entropy_lambda" => Objective::CrossEntropyLambda,
            "lambdarank" | "rank_xendcg" | "custom" | "none" | "" => Objective::Identity,
            other => return Err(Error::Unsupported(format!("objective `{other}`"))),
        })
    }
}

/// One tree. Internal nodes are numbered from 0 (the root); a child index `c < 0` is leaf `!c`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tree {
    pub split_feature: Vec<u32>, pub threshold: Vec<f64>, pub decision_type: Vec<u8>, pub left: Vec<i32>, pub right: Vec<i32>, pub leaf_value: Vec<f64>,
    pub internal_count: Vec<u64>, pub leaf_count: Vec<u64>, pub cat_boundaries: Vec<u32>, pub cat_threshold: Vec<u32>
}

impl Tree {
    /// Number of internal nodes.
    pub fn num_nodes(&self) -> usize { self.split_feature.len() }

    /// Training rows that reached child `c` (0 when the model has no counts).
    pub fn count(&self, c: i32) -> u64 {
        let v = if c < 0 { self.leaf_count.get(!c as usize) } else { self.internal_count.get(c as usize) };
        v.copied().unwrap_or(0)
    }

    pub fn missing_type(&self, n: usize) -> MissingType {
        match (self.decision_type[n] >> 2) & 3 { 0 => MissingType::None, 1 => MissingType::Zero, _ => MissingType::NaN }
    }

    pub fn default_left(&self, n: usize) -> bool { self.decision_type[n] & 2 != 0 }

    pub fn is_categorical(&self, n: usize) -> bool { self.decision_type[n] & 1 != 0 }

    /// Bitset of categorical node `n`: category `v` goes left when bit `v % 32` of word `v / 32` is set.
    pub fn bitset(&self, n: usize) -> &[u32] {
        let i = self.threshold[n] as usize;
        &self.cat_threshold[self.cat_boundaries[i] as usize..self.cat_boundaries[i + 1] as usize]
    }

    /// Whether a NaN input goes left at node `n`.
    pub fn nan_left(&self, n: usize) -> bool {
        if self.is_categorical(n) { return false; }
        match self.missing_type(n) { MissingType::NaN | MissingType::Zero => self.default_left(n), MissingType::None => 0.0 <= self.threshold[n] }
    }

    /// Number of leaves below each internal node.
    pub fn n_leaves(&self) -> Vec<u32> {
        fn go(t: &Tree, c: i32, nl: &mut [u32]) -> u32 {
            if c < 0 { return 1; }
            let k = go(t, t.left[c as usize], nl) + go(t, t.right[c as usize], nl);
            nl[c as usize] = k;
            k
        }
        let mut nl = vec![0; self.num_nodes()];
        if self.num_nodes() > 0 { go(self, 0, &mut nl); }
        nl
    }
}

/// A parsed LightGBM model.
#[derive(Clone, Debug, PartialEq)]
pub struct Forest {
    pub trees: Vec<Tree>,
    pub num_feature: usize,
    pub num_class: usize,
    /// Trees per boosting iteration, which is also the number of outputs per row.
    pub trees_per_iteration: usize,
    pub objective: Objective,
    pub average_output: bool,
    pub feature_names: Vec<String>,
    /// Whether each feature is categorical, from `feature_infos`.
    pub categorical: Vec<bool>,
    /// The JSON after `pandas_categorical:`, or `None` for `null`.
    pub pandas_categorical: Option<String>,
    /// The model text as given.
    pub text: String,
}

fn list<T: std::str::FromStr>(kv: &HashMap<&str, &str>, key: &str) -> Result<Vec<T>, Error> {
    let s = kv.get(key).copied().unwrap_or("");
    s.split_whitespace().map(|x| x.parse().map_err(|_| parse_err(format!("bad value `{x}` in `{key}`")))).collect()
}

fn scalar<T: std::str::FromStr>(kv: &HashMap<&str, &str>, key: &str) -> Result<Option<T>, Error> {
    kv.get(key).map(|v| v.trim().parse().map_err(|_| parse_err(format!("bad value `{v}` for `{key}`")))).transpose()
}

fn parse_tree(kv: &HashMap<&str, &str>, categorical: &[bool]) -> Result<Tree, Error> {
    if kv.get("is_linear").is_some_and(|v| v.trim() != "0") { return Err(Error::Unsupported("linear trees".into())); }
    let t = Tree { split_feature: list(kv, "split_feature")?, threshold: list(kv, "threshold")?, decision_type: list(kv, "decision_type")?,
        left: list(kv, "left_child")?, right: list(kv, "right_child")?, leaf_value: list(kv, "leaf_value")?, internal_count: list(kv, "internal_count")?,
        leaf_count: list(kv, "leaf_count")?, cat_boundaries: list(kv, "cat_boundaries")?, cat_threshold: list(kv, "cat_threshold")? };
    let n = t.num_nodes();
    let leaves = t.leaf_value.len();
    if leaves != n + 1 { return Err(parse_err(format!("{leaves} leaves for {n} nodes"))); }
    for (name, len) in
        [("threshold", t.threshold.len()), ("decision_type", t.decision_type.len()), ("left_child", t.left.len()), ("right_child", t.right.len())] {
            if len != n { return Err(parse_err(format!("`{name}` has {len} entries for {n} nodes"))); }
        }
    for i in 0..n {
        for c in [t.left[i], t.right[i]] {
            let ok = if c < 0 { (!c as usize) < leaves } else { (c as usize) < n && c as usize > i };
            if !ok { return Err(parse_err(format!("node {i} has invalid child {c}"))); }
        }
        let f = t.split_feature[i] as usize;
        if f >= categorical.len() { return Err(parse_err(format!("node {i} splits on feature {f}"))); }
        if (t.decision_type[i] >> 2) & 3 == 3 { return Err(parse_err(format!("node {i} has decision type {}", t.decision_type[i]))); }
        if t.is_categorical(i) {
            let k = t.threshold[i];
            if !(k >= 0.0 && (k as usize) + 1 < t.cat_boundaries.len()) || k.fract() != 0.0 {
                return Err(parse_err(format!("categorical node {i} has threshold {k}")));
            }
            let (a, b) = (t.cat_boundaries[k as usize] as usize, t.cat_boundaries[k as usize + 1] as usize);
            if a > b || b > t.cat_threshold.len() { return Err(parse_err(format!("categorical node {i} has bitset {a}..{b}"))); }
        }
        if t.is_categorical(i) != categorical[f] {
            return Err(Error::Unsupported(format!(
                "node {i} splits feature {f} as {}, but `feature_infos` marks it {}", if t.is_categorical(i) { "categorical" } else { "numerical" },
                if categorical[f] { "categorical" } else { "numerical" },
            )));
        }
    }
    Ok(t)
}

impl Forest {
    /// Parses LightGBM's text model format, as written by `Booster.save_model`.
    pub fn parse(text: &str) -> Result<Forest, Error> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some("tree") { return Err(parse_err("the first line is not `tree`")); }
        let mut header: HashMap<&str, &str> = HashMap::new();
        let mut average_output = false;
        let mut tree_kvs: Vec<HashMap<&str, &str>> = vec![];
        let mut ended = false;
        for line in lines.by_ref() {
            let line = line.trim();
            if line == "end of trees" { ended = true; break; }
            if line.starts_with("Tree=") { tree_kvs.push(HashMap::new()); }
            let target = match tree_kvs.last_mut() { Some(kv) => kv, None => &mut header };
            match line.split_once('=') {
                Some((k, v)) => { target.insert(k, v); } None if line == "average_output" && tree_kvs.is_empty() => average_output = true, None => {}
            }
        }
        if !ended { return Err(parse_err("no `end of trees` line")); }
        let pandas_categorical = lines.filter_map(|l| l.trim().strip_prefix("pandas_categorical:")).next_back().filter(|v| *v != "null").map(str::to_string);

        let num_feature = scalar::<usize>(&header, "max_feature_idx")?.ok_or_else(|| parse_err("no `max_feature_idx`"))? + 1;
        let num_class = scalar::<usize>(&header, "num_class")?.unwrap_or(1);
        let trees_per_iteration = scalar::<usize>(&header, "num_tree_per_iteration")?.unwrap_or(num_class);
        let objective = match header.get("objective") { Some(s) => Objective::parse(s)?, None => Objective::Identity };
        let feature_names: Vec<String> = match header.get("feature_names") {
            Some(s) => s.split_whitespace().map(str::to_string).collect(), None => (0..num_feature).map(|i| format!("Column_{i}")).collect()
        };
        let categorical: Vec<bool> = match header.get("feature_infos") {
            Some(s) => s.split_whitespace().map(|i| !i.starts_with('[') && i != "none").collect(), None => vec![false; num_feature]
        };
        if feature_names.len() != num_feature || categorical.len() != num_feature {
            return Err(parse_err(format!("{num_feature} features but {} names and {} infos", feature_names.len(), categorical.len())));
        }
        let trees = tree_kvs.iter().map(|kv| parse_tree(kv, &categorical)).collect::<Result<Vec<_>, _>>()?;
        if trees_per_iteration == 0 || trees.len() % trees_per_iteration != 0 {
            return Err(parse_err(format!("{} trees for {trees_per_iteration} trees per iteration", trees.len())));
        }
        Ok(Forest { trees, num_feature, num_class, trees_per_iteration, objective, average_output, feature_names, categorical, pandas_categorical,
            text: text.to_string() })
    }

    /// Number of boosting iterations.
    pub fn num_iterations(&self) -> usize { self.trees.len() / self.trees_per_iteration }
}
