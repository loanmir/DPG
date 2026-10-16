import networkx as nx
import pandas as pd
import pytest
from networkx.readwrite import json_graph

from categorical import cat_grouping
from categorical import cat_grouping_split_conjunction as conjunction
from categorical import dpg_equivalence
from categorical.onehot import build_onehot_map, categorical_columns, split_one_hot_column


@pytest.fixture
def frame():
    return pd.DataFrame({
        "job": ["self_employed", "admin", "admin", "self_employed", "admin"],
        "has_loan": [0, 1, 0, 1, 1],
        "children": pd.Series(["1", "2", "more", "1", "2"], dtype=object),
        "Maintenance_Hours": [0, 3, 5, 7, 9],
    })


@pytest.fixture
def onehot_map(frame):
    return build_onehot_map(frame)


def _structure(edges, onehot_map=None):
    graph = nx.DiGraph()
    labels = {}
    for a, b in edges:
        labels[a], labels[b] = a, b
        graph.add_edge(a, b, weight=1.0)
    structure = {
        "nodes": [{"id": k, "label": v} for k, v in labels.items()],
        "graph": json_graph.node_link_data(graph),
    }
    if onehot_map is not None:
        structure["onehot_map"] = {d: [c, v] for d, (c, v) in onehot_map.items()}
    return structure


def test_categorical_columns_by_dtype():
    df = pd.DataFrame({
        "obj": pd.Series(["a", "b"], dtype=object),
        "string": pd.Series(["a", "b"], dtype=pd.StringDtype()),  # what pandas 3 gives text
        "cat": pd.Series(["a", "b"], dtype="category"),
        "int": [1, 2],
        "float": [1.0, 2.5],
        "bool": [True, False],  # get_dummies passes bool columns through
    })
    assert categorical_columns(df) == ["obj", "string", "cat"]


def test_categorical_columns_nan_first_row():
    df = pd.DataFrame({"text": [None, "a", "b"], "num": [1, 2, 3]})
    assert categorical_columns(df) == ["text"]


def test_categorical_columns_all_numeric():
    assert categorical_columns(pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})) == []


def test_categorical_columns_name_clash():
    df = pd.DataFrame({"a": ["b", "c"], "a_b": [1, 2]})
    with pytest.raises(ValueError):
        categorical_columns(df)


def test_load_categorical_domains(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "job": ["admin", "self_employed", "admin", None],
        "age": [30, 40, 50, 60],
        "target": ["yes", "no", "yes", "no"],
    }).to_csv(csv, index=False)
    expected = {
        "job": frozenset({"admin", "self_employed"}),
        "target": frozenset({"yes", "no"}),
    }
    assert dpg_equivalence.load_categorical_domains(str(csv), drop_columns=["id"]) == expected

    df = pd.read_csv(csv).drop(columns=["id"])
    assert dpg_equivalence._domains_from_frame(df) == expected
    # Simulates pandas 3, where text columns get a string dtype instead of object.
    assert dpg_equivalence._domains_from_frame(df.convert_dtypes()) == expected


def test_build_onehot_map(onehot_map):
    assert onehot_map == {
        "job_self_employed": ("job", "self_employed"),
        "job_admin": ("job", "admin"),
        "children_1": ("children", "1"),
        "children_2": ("children", "2"),
        "children_more": ("children", "more"),
    }


def test_build_onehot_map_matches_get_dummies(frame, onehot_map):
    numeric = {"has_loan", "Maintenance_Hours"}
    assert set(pd.get_dummies(frame, drop_first=False).columns) == set(onehot_map) | numeric


def test_split_one_hot_column_with_map(onehot_map):
    assert split_one_hot_column("job_self_employed", onehot_map) == ("job", "self_employed")
    assert split_one_hot_column("has_loan", onehot_map) is None
    assert split_one_hot_column("Maintenance_Hours", onehot_map) is None
    assert split_one_hot_column("children_1", onehot_map) == ("children", "1")


def test_split_one_hot_column_with_json_map(onehot_map):
    json_map = {d: [c, v] for d, (c, v) in onehot_map.items()}
    assert split_one_hot_column("job_self_employed", json_map) == ("job", "self_employed")


def test_split_one_hot_column_legacy_guess():
    assert split_one_hot_column("job_self_employed") == ("job_self", "employed")
    assert split_one_hot_column("has_loan") == ("has", "loan")
    assert split_one_hot_column("children_1") is None


def test_build_onehot_map_collision():
    df = pd.DataFrame({"a_b": ["c", "c"], "a": ["b_c", "x"]})
    with pytest.raises(ValueError):
        build_onehot_map(df)


def test_build_onehot_map_collision_with_numeric():
    df = pd.DataFrame({"a_b": [1, 2], "a": ["b", "x"]})
    with pytest.raises(ValueError):
        build_onehot_map(df)


def test_build_onehot_map_warns_on_grammar_chars(capsys):
    build_onehot_map(pd.DataFrame({"x": ["a,b", "c"]}))
    assert "[warn]" in capsys.readouterr().out


def test_build_label_map_uses_onehot_map():
    onehot_map = {
        "job_self_employed": ("job", "self_employed"),
        "MTRANS_Public_Transportation": ("MTRANS", "Public_Transportation"),
    }
    structure = _structure(
        [("has_loan > 0.5", "job_self_employed <= 0.5"),
         ("job_self_employed <= 0.5", "MTRANS_Public_Transportation > 0.5"),
         ("MTRANS_Public_Transportation > 0.5", "Class 1")],
        onehot_map,
    )
    new_labels, _ = cat_grouping._build_label_map(structure)
    assert new_labels["has_loan > 0.5"] == "has_loan > 0.5"
    assert new_labels["job_self_employed <= 0.5"] == "job NOT IN {self_employed}"
    assert new_labels["MTRANS_Public_Transportation > 0.5"] == "MTRANS IN {Public_Transportation}"


def test_build_label_map_warns_without_map(capsys):
    cat_grouping._build_label_map(_structure([("a_X > 0.5", "Class 1")]))
    assert "no onehot_map" in capsys.readouterr().out


def test_conjunction_does_not_merge_numeric_node():
    onehot_map = {"Weather_Fog": ("Weather", "Fog")}
    structure = _structure(
        [("Maintenance_Hours > 0.5", "Weather_Fog > 0.5"), ("Weather_Fog > 0.5", "Class 1")],
        onehot_map,
    )
    label_map, _ = cat_grouping._build_label_map(structure)
    labels, _ = conjunction._split_then_merge(structure, label_map)
    assert "Maintenance_Hours > 0.5" in labels.values()
    assert "Weather IN {Fog}" in labels.values()


def test_canonical_constraints_numeric_with_map():
    onehot_map = {"job_self_employed": ("job", "self_employed")}
    constraints = dpg_equivalence.canonical_constraints(["has_loan > 0.5"], onehot_map=onehot_map)
    assert constraints == frozenset({("num", "has_loan", ">", 0.5)})
    constraints = dpg_equivalence.canonical_constraints(["job_self_employed > 0.5"], onehot_map=onehot_map)
    assert constraints == frozenset({("cat", "job", "IN", frozenset({"self_employed"}))})
