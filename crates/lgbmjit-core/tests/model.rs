mod common;
use common::{load, model_names, model_text};
use lgbmjit_core::model::{Error, Forest, Objective};

/// The bundled models parse with the header values, class layout, categorical splits and leaves LightGBM wrote.
#[test]
fn parses_bundled_models() {
    for name in model_names() {
        let f = load(&name);
        assert!(!f.trees.is_empty(), "{name}");
        assert_eq!(f.categorical.len(), f.num_feature, "{name}");
        assert_eq!(f.trees.len() % f.trees_per_iteration, 0, "{name}");
        assert_eq!(f.text, model_text(&name));
        for t in &f.trees {
            for n in (0..t.num_nodes()).filter(|&n| t.is_categorical(n)) {
                assert!(f.categorical[t.split_feature[n] as usize] && !t.bitset(n).is_empty(), "{name}");
            }
        }
    }
    let nyc = load("NYC_taxi");
    assert_eq!((nyc.trees.len(), nyc.num_feature, nyc.trees_per_iteration), (100, 5, 1));
    assert_eq!((nyc.objective, nyc.average_output, nyc.pandas_categorical), (Objective::Identity, false, None));
    let mc = load("multiclass");
    assert_eq!((mc.num_class, mc.trees_per_iteration, mc.objective), (3, 3, Objective::Softmax));
    assert!(load("airline").trees.iter().any(|t| (0..t.num_nodes()).any(|n| t.is_categorical(n))));
    let single = load("single_leaf");
    assert!(single.trees[0].split_feature.is_empty());
    assert_eq!(single.trees[0].leaf_value, vec![2.0]);
}

#[test]
fn parses_objective_tokens() {
    assert_eq!(Objective::parse("binary sigmoid:1").unwrap(), Objective::Sigmoid(1.0));
    assert_eq!(Objective::parse("regression sqrt").unwrap(), Objective::Sqrt);
    assert_eq!(Objective::parse("regression_l1 sqrt").unwrap(), Objective::Sqrt);
    assert_eq!(Objective::parse("huber").unwrap(), Objective::Identity);
    assert_eq!(Objective::parse("multiclassova num_class:3 sigmoid:2").unwrap(), Objective::OvaSigmoid(2.0));
    assert_eq!(Objective::parse("multiclass num_class:3").unwrap(), Objective::Softmax);
    assert_eq!(Objective::parse("tweedie").unwrap(), Objective::Exp);
    assert_eq!(Objective::parse("cross_entropy").unwrap(), Objective::CrossEntropy);
    assert_eq!(Objective::parse("cross_entropy_lambda").unwrap(), Objective::CrossEntropyLambda);
    assert_eq!(Objective::parse("lambdarank").unwrap(), Objective::Identity);
    assert!(matches!(Objective::parse("my_objective"), Err(Error::Unsupported(_))));
    let text: String = model_text("NYC_taxi").lines().filter(|l| !l.starts_with("objective=")).map(|l| format!("{l}\n")).collect();
    assert_eq!(Forest::parse(&text).unwrap().objective, Objective::Identity);
}

/// Linear trees and categorical splits on features marked numerical are unsupported; other text is not a model.
#[test]
fn rejects_unsupported_and_invalid_models() {
    let linear = model_text("single_leaf").replace("is_linear=0", "is_linear=1");
    assert!(matches!(Forest::parse(&linear), Err(Error::Unsupported(_))));
    let numerical_infos: String = model_text("airline").lines().map(|l| match l.strip_prefix("feature_infos=") {
        Some(v) => format!("feature_infos={}\n", vec!["[0:1]"; v.split(' ').count()].join(" ")), None => format!("{l}\n")
    }).collect();
    assert!(matches!(Forest::parse(&numerical_infos), Err(Error::Unsupported(_))));
    assert!(matches!(Forest::parse("hello"), Err(Error::Parse(_))));
}
