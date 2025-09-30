<div align="center">
  <picture>
    <img alt="Infera Logo" src="logo.svg" height="25%" width="25%">
  </picture>
<br>

<h2>Infera</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/CogitatorTech/infera/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/infera/actions/workflows/tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/CogitatorTech/infera?label=coverage&style=flat&labelColor=282c34&logo=codecov)](https://codecov.io/gh/CogitatorTech/infera)
[![Code Quality](https://img.shields.io/codefactor/grade/github/CogitatorTech/infera?label=quality&style=flat&labelColor=282c34&logo=codefactor)](https://www.codefactor.io/repository/github/CogitatorTech/infera)
[![Examples](https://img.shields.io/badge/examples-view-green?style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/infera/tree/main/docs/examples)
[![Docs](https://img.shields.io/badge/docs-view-blue?style=flat&labelColor=282c34&logo=read-the-docs)](https://github.com/CogitatorTech/infera/tree/main/docs)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?style=flat&labelColor=282c34&logo=open-source-initiative)](https://github.com/CogitatorTech/infera)

In-Database Machine Learning for DuckDB

</div>

---

Infera is DuckDB extension that allows you use machine learning (ML) models directly in SQL queries to perform inference
on data stored in DuckDB tables.
It is developed in Rust and uses [Tract](https://github.com/snipsco/tract) as the backend inference engine.
Infera supports loading and running models in [ONNX](https://onnx.ai/) format.
Check out the [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo) repositors on Hugging Face for a very large
collection of ready-to-use models that can be used with Infera.

### Motivation

In a conventional data science workflow, when data is stored in a database, it is not normally possible to use ML models
directly on the data.
Users need to move the data out of the database first (for example, export it to a CSV file), load the data into a
Python or R environment, run the model there, and then import the results back into the database.
This process is time-consuming and inefficient.
Infera aims to solve this problem by letting users to run ML models directly in SQL queries inside the database.
It simplifies the workflow and speeds up the process for users, and eliminates the need for moving data around.

### Features

- Adds ML inference as first-class citizens in SQL queries.
- Supports loading and using local as well as remote models.
- Supports using ML models in ONNX format with a simple and flexible API.
- Supports performing inference on table columns or raw `BLOB` (tensor) data.
- Supports both single-value and multi-value model outputs.
- Supports autoloading all models from a specified directory.
- Thread-safe, fast, and memory-efficient.

See the [ROADMAP.md](ROADMAP.md) for the list of implemented and planned features.

> [!IMPORTANT]
> Infera is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/CogitatorTech/infera/issues) to report bugs or request features.

---

### Quickstart

```sql
-- 1. Load the extension
LOAD
infera;

-- 2. Load a simple linear model from a remote URL
SELECT infera_load_model('linear_model',
                         'https://github.com/CogitatorTech/infera/raw/refs/heads/main/test/models/linear.onnx');

-- 3. Run a prediction using a very simple linear model
-- Model: y = 2*x1 - 1*x2 + 0.5*x3 + 0.25
SELECT infera_predict('linear_model', 1.0, 2.0, 3.0);
-- Expected output: 1.75

-- 4. Unload the model when we're done with it
SELECT infera_unload_model('linear_model');
````

---

### Documentation

Check out the [docs](docs/README.md) directory for the API documentation, how to build Infera from source, and more.

#### Examples

Check out the [examples](docs/examples) directory for SQL scripts that show how to use Infera.

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

Infera is available under either of the following licenses:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

### Acknowledgements

* The logo is from [here](https://www.svgrepo.com/svg/499306/overmind) with some modifications.
