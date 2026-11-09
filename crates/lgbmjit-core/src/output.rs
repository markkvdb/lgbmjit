//! Output transforms applied after the raw scores are summed.
use crate::model::{Forest, Objective};
use crate::reference::IterRange;

fn sigmoid(s: f64, x: f64) -> f64 { 1.0 / (1.0 + (-s * x).exp()) }

/// Applies `average_output` (skipped when `raw_score`) and the objective's transform to rows of `out`, each `forest.trees_per_iteration` long.
pub fn finish_rows(forest: &Forest, range: IterRange, raw_score: bool, out: &mut [f64]) {
    if raw_score { return; }
    let k = forest.trees_per_iteration;
    if forest.average_output {
        let n = range.count as f64;
        out.iter_mut().for_each(|x| *x /= n);
    }
    match forest.objective {
        Objective::Identity => {}
        Objective::Sqrt => out.iter_mut().for_each(|x| {
            let sign = ((*x > 0.0) as i32 - (*x < 0.0) as i32) as f64;
            *x = sign * *x * *x;
        }),
        Objective::Exp => out.iter_mut().for_each(|x| *x = x.exp()),
        Objective::Sigmoid(s) | Objective::OvaSigmoid(s) => out.iter_mut().for_each(|x| *x = sigmoid(s, *x)),
        Objective::CrossEntropy => out.iter_mut().for_each(|x| *x = sigmoid(1.0, *x)),
        Objective::CrossEntropyLambda => out.iter_mut().for_each(|x| *x = x.exp().ln_1p()),
        Objective::Softmax => out.chunks_exact_mut(k).for_each(|row| {
            // std::max(x, m) in LightGBM's Softmax
            let m = row.iter().skip(1).fold(row[0], |m, &x| if x < m { m } else { x });
            let mut sum = 0.0;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            row.iter_mut().for_each(|x| *x /= sum);
        }),
    }
}
