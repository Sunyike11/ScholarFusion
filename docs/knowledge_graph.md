# Knowledge Graph

The graph is stored in SQLite, not Neo4j.

Node types:

- Paper
- Method
- Dataset
- Metric
- BodyModel
- Representation
- Figure

Relations:

- PROPOSES
- USES_BODY_MODEL
- USES_REPRESENTATION
- EVALUATED_ON
- REPORTS_METRIC
- ILLUSTRATES

Graph edges are generated from Paper Card fields and figure evidence. Related-work summaries are template-based and cite graph evidence paths.
