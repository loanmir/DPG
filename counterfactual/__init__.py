"""Counterfactual explanation module for DPG.

Provides counterfactual generation, constraint extraction, scoring, and
visualisation on top of Decision Predicate Graph models.
"""

import importlib as _importlib

__all__ = [
    # boundary_analyzer
    "BoundaryAnalyzer",
    # constraint_parser
    "ConstraintParser",
    # constraint_scorer
    "compare_constraints",
    "compute_constraint_score",
    "compute_constraint_score_from_file",
    "load_constraints_from_json",
    # constraint_validator
    "ConstraintValidator",
    # counterfactual_explainer
    "CounterFactualExplainer",
    # counterfactual_metrics
    "avg_nbr_changes",
    "avg_nbr_changes_per_cf",
    "avg_nbr_violations",
    "avg_nbr_violations_per_cf",
    "categorical_distance",
    "categorical_diversity",
    "continuous_distance",
    "continuous_diversity",
    "count_diversity",
    "count_diversity_all",
    "distance_l2j",
    "distance_mh",
    "diversity_l2j",
    "diversity_mh",
    "mad_cityblock",
    "nbr_actionable_cf",
    "nbr_changes_per_cf",
    "nbr_valid_actionable_cf",
    "nbr_valid_cf",
    "nbr_violations_per_cf",
    "perc_actionable_cf",
    "perc_valid_actionable_cf",
    "perc_valid_cf",
    "plausibility",
    # counterfactual_model
    "CounterFactualModel",
    # counterfactual_visualizer
    "plot_constraints",
    "plot_explainer_summary",
    "plot_fitness",
    "plot_fitness_std",
    "plot_pca_loadings",
    "plot_pca_with_counterfactual",
    "plot_pca_with_counterfactuals",
    "plot_pca_with_counterfactuals_clean",
    "plot_pca_with_counterfactuals_comparison",
    "plot_sample_and_counterfactual_comparison",
    "plot_sample_and_counterfactual_comparison_combined",
    "plot_sample_and_counterfactual_comparison_simple",
    "plot_sample_and_counterfactual_heatmap",
    "set_default_style",
    # fitness_calculator
    "FitnessCalculator",
    # heuristic_runner
    "DistanceBasedHOF",
    "HeuristicRunner",
    # sample_generator
    "SampleGenerator",
]

# Mapping from public name to (submodule, attribute) for lazy imports.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BoundaryAnalyzer": (".boundary_analyzer", "BoundaryAnalyzer"),
    "ConstraintParser": (".constraint_parser", "ConstraintParser"),
    "compare_constraints": (".constraint_scorer", "compare_constraints"),
    "compute_constraint_score": (".constraint_scorer", "compute_constraint_score"),
    "compute_constraint_score_from_file": (".constraint_scorer", "compute_constraint_score_from_file"),
    "load_constraints_from_json": (".constraint_scorer", "load_constraints_from_json"),
    "ConstraintValidator": (".constraint_validator", "ConstraintValidator"),
    "CounterFactualExplainer": (".counterfactual_explainer", "CounterFactualExplainer"),
    "CounterFactualModel": (".counterfactual_model", "CounterFactualModel"),
    "FitnessCalculator": (".fitness_calculator", "FitnessCalculator"),
    "DistanceBasedHOF": (".heuristic_runner", "DistanceBasedHOF"),
    "HeuristicRunner": (".heuristic_runner", "HeuristicRunner"),
    "SampleGenerator": (".sample_generator", "SampleGenerator"),
}

# Metrics and visualiser functions — mapped in bulk.
for _name in (
    "avg_nbr_changes", "avg_nbr_changes_per_cf", "avg_nbr_violations",
    "avg_nbr_violations_per_cf", "categorical_distance", "categorical_diversity",
    "continuous_distance", "continuous_diversity", "count_diversity",
    "count_diversity_all", "distance_l2j", "distance_mh", "diversity_l2j",
    "diversity_mh", "mad_cityblock", "nbr_actionable_cf", "nbr_changes_per_cf",
    "nbr_valid_actionable_cf", "nbr_valid_cf", "nbr_violations_per_cf",
    "perc_actionable_cf", "perc_valid_actionable_cf", "perc_valid_cf",
    "plausibility",
):
    _LAZY_IMPORTS[_name] = (".counterfactual_metrics", _name)

for _name in (
    "plot_constraints", "plot_explainer_summary", "plot_fitness",
    "plot_fitness_std", "plot_pca_loadings", "plot_pca_with_counterfactual",
    "plot_pca_with_counterfactuals", "plot_pca_with_counterfactuals_clean",
    "plot_pca_with_counterfactuals_comparison",
    "plot_sample_and_counterfactual_comparison",
    "plot_sample_and_counterfactual_comparison_combined",
    "plot_sample_and_counterfactual_comparison_simple",
    "plot_sample_and_counterfactual_heatmap", "set_default_style",
):
    _LAZY_IMPORTS[_name] = (".counterfactual_visualizer", _name)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        # Cache on the module so __getattr__ is not called again.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
