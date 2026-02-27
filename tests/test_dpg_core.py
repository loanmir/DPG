"""
Tests for DPG core pipeline: DecisionPredicateGraph construction,
NetworkX conversion, and graph structure validation.

Uses the Iris dataset with a fixed random seed so that expected node/edge
counts and metric ranges are deterministic and reproducible.
"""

import re

import networkx as nx
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

from dpg.core import DecisionPredicateGraph, DPGError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 160898


@pytest.fixture(scope="module")
def iris_split():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=SEED
    )
    target_names = np.unique(iris.target).astype(str).tolist()
    return X_train, X_test, y_train, y_test, iris.feature_names, target_names


@pytest.fixture(scope="module")
def iris_rf(iris_split):
    X_train, _, y_train, _, _, _ = iris_split
    model = RandomForestClassifier(n_estimators=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def iris_dpg(iris_rf, iris_split):
    """Build a DPG for Iris and return (dpg_graph, nodes_list, dot)."""
    X_train, _, _, _, feature_names, target_names = iris_split
    dpg = DecisionPredicateGraph(
        model=iris_rf,
        feature_names=feature_names,
        target_names=target_names,
    )
    dot = dpg.fit(X_train)
    dpg_graph, nodes_list = dpg.to_networkx(dot)
    return dpg_graph, nodes_list, dot


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Validate the DPG graph has the expected topology for Iris."""

    def test_graph_is_directed(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert isinstance(dpg_graph, nx.DiGraph)

    def test_exact_node_count(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert dpg_graph.number_of_nodes() == 31

    def test_exact_edge_count(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert dpg_graph.number_of_edges() == 51

    def test_nodes_list_matches_graph(self, iris_dpg):
        """nodes_list (non-edge entries) should match the graph node set."""
        dpg_graph, nodes_list, _ = iris_dpg
        node_ids = {n[0] for n in nodes_list if "->" not in n[0]}
        assert node_ids == set(dpg_graph.nodes())

    def test_all_edges_have_weight(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        for u, v, data in dpg_graph.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 1

    def test_class_nodes_are_sinks(self, iris_dpg):
        """Class nodes should have out-degree 0 (terminal/absorbing)."""
        dpg_graph, nodes_list, _ = iris_dpg
        class_node_ids = [n[0] for n in nodes_list if "Class" in n[1]]
        for node_id in class_node_ids:
            assert dpg_graph.out_degree(node_id) == 0

    def test_three_class_nodes_present(self, iris_dpg):
        _, nodes_list, _ = iris_dpg
        class_labels = sorted(
            {n[1] for n in nodes_list if n[1].startswith("Class ")}
        )
        assert class_labels == ["Class 0", "Class 1", "Class 2"]

    def test_predicate_node_label_format(self, iris_dpg):
        """Non-class node labels should be feature predicates like 'feat <= val' or 'feat > val'."""
        _, nodes_list, _ = iris_dpg
        predicate_re = re.compile(r"^.+\s*(<=|>)\s*[\d.eE+-]+$")
        for node_id, label in nodes_list:
            if "->" in node_id or label.startswith("Class "):
                continue
            assert predicate_re.match(label), f"Unexpected predicate format: {label!r}"

    def test_graph_is_weakly_connected(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert nx.is_weakly_connected(dpg_graph)


# ---------------------------------------------------------------------------
# Dot / Graphviz output tests
# ---------------------------------------------------------------------------


class TestDotOutput:
    def test_dot_is_graphviz_digraph(self, iris_dpg):
        import graphviz

        _, _, dot = iris_dpg
        assert isinstance(dot, graphviz.Digraph)

    def test_dot_source_contains_class_labels(self, iris_dpg):
        _, _, dot = iris_dpg
        source = dot.source
        for cls in ["Class 0", "Class 1", "Class 2"]:
            assert cls in source


# ---------------------------------------------------------------------------
# Initialization validation
# ---------------------------------------------------------------------------


class TestDPGInitValidation:
    def test_rejects_non_ensemble_model(self, iris_split):
        _, _, _, _, feature_names, _ = iris_split
        with pytest.raises(DPGError, match="tree-based ensemble"):
            DecisionPredicateGraph(model="not_a_model", feature_names=feature_names)

    def test_rejects_empty_feature_names(self, iris_rf):
        with pytest.raises(DPGError, match="Feature names cannot be empty"):
            DecisionPredicateGraph(model=iris_rf, feature_names=[])

    def test_accepts_custom_dpg_config(self, iris_rf, iris_split):
        _, _, _, _, feature_names, target_names = iris_split
        custom_config = {
            "dpg": {
                "default": {
                    "perc_var": 0.01,
                    "decimal_threshold": 3,
                    "n_jobs": 1,
                }
            }
        }
        dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config=custom_config,
        )
        assert dpg.perc_var == 0.01
        assert dpg.decimal_threshold == 3
        assert dpg.n_jobs == 1


# ---------------------------------------------------------------------------
# Different ensemble models
# ---------------------------------------------------------------------------


class TestMultipleModels:
    def test_extra_trees_produces_valid_dpg(self, iris_split):
        X_train, _, y_train, _, feature_names, target_names = iris_split
        model = ExtraTreesClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        dot = dpg.fit(X_train)
        graph, nodes = dpg.to_networkx(dot)
        assert graph.number_of_nodes() > 3
        assert graph.number_of_edges() > 3
        class_labels = {n[1] for n in nodes if n[1].startswith("Class ")}
        assert len(class_labels) == 3


class TestDifferentDataset:
    """Validate DPG works on Wine (more features, same 3-class setup)."""

    def test_wine_graph_structure(self):
        wine = load_wine()
        X_train, _, y_train, _ = train_test_split(
            wine.data, wine.target, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        target_names = np.unique(wine.target).astype(str).tolist()
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=wine.feature_names,
            target_names=target_names,
        )
        dot = dpg.fit(X_train)
        graph, nodes = dpg.to_networkx(dot)

        assert graph.number_of_nodes() == 87
        assert graph.number_of_edges() == 126
        class_labels = sorted({n[1] for n in nodes if n[1].startswith("Class ")})
        assert class_labels == ["Class 0", "Class 1", "Class 2"]
