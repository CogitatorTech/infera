<div align="center">
  <picture>
    <img alt="Infera Logo" src="logo.svg" height="25%" width="25%">
  </picture>
<br>

<h2>Infera</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/CogitatorTech/infera/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/infera/actions/workflows/tests.yml)
[![Code Quality](https://img.shields.io/codefactor/grade/github/CogitatorTech/infera?label=quality&style=flat&labelColor=282c34&logo=codefactor)](https://www.codefactor.io/repository/github/CogitatorTech/infera)
[![Examples](https://img.shields.io/badge/examples-view-green?style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/infera/tree/main/docs/examples)
[![Docs](https://img.shields.io/badge/docs-view-blue?style=flat&labelColor=282c34&logo=read-the-docs)](https://github.com/CogitatorTech/infera/tree/main/docs)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?style=flat&labelColor=282c34&logo=open-source-initiative)](https://github.com/CogitatorTech/infera)

In-Database Machine Learning for DuckDB

</div>

---

Infera is a DuckDB extension that allows you to use machine learning (ML) models directly in SQL queries to perform
inference on data stored in DuckDB tables.
It is developed in Rust and uses [Tract](https://github.com/snipsco/tract) as the backend inference engine.
Infera supports loading and running models in [ONNX](https://onnx.ai/) format.
Check out the [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo) repository on Hugging Face for a large
collection of ready-to-use models that can be used with Infera.

### Motivation

In a conventional data science workflow, when data is stored in a database, it is not typically possible to use ML
models directly on the data.
Users need to move the data out of the database first (for example, export it to a CSV file) and load the data into a
Python or R environment, run the model there, and then import the results back into the database.
This process is time-consuming and inefficient.
Infera aims to solve this problem by letting users run ML models directly in SQL queries inside the database.
It simplifies the workflow and speeds up the process for users, and eliminates the need for moving data around.

### Features

- Adds ML inference as a first-class citizen in SQL queries.
- Supports loading and using local as well as remote models.
- Supports using ML models in ONNX format with a simple and flexible API.
- Supports performing inference on table columns or raw tensor data.
- Supports both single-value and multi-value model outputs.
- Supports autoloading all models from a specified directory.
- Thread-safe, fast, and memory-efficient.

See the [ROADMAP.md](ROADMAP.md) for the list of implemented and planned features.

> [!IMPORTANT]
> Infera is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/CogitatorTech/infera/issues) to report bugs or request features.

---

### Quickstart

1. Clone the repository and build the Infera extension from source:

```bash
git clone --recursive https://github.com/CogitatorTech/infera.git
cd infera

# This might take a while to run
make release
```

2. Start DuckDB shell (with Infera statically linked to it):

```bash
./build/release/duckdb
```

3. Run the following SQL commands in the shell to try Infera out:

```sql
-- Normally, we need to load the extension first,
-- but the `duckdb` binary that we built in the previous step
-- already has Infera statically linked to it.
-- So, we don't need to load the extension explicitly.

-- 1. Load a simple linear model from a remote URL
select infera_load_model('linear_model',
                         'https://github.com/CogitatorTech/infera/raw/refs/heads/main/test/models/linear.onnx');

-- 2. Run a prediction using a very simple linear model
-- Model: y = 2*x1 - 1*x2 + 0.5*x3 + 0.25
select infera_predict('linear_model', 1.0, 2.0, 3.0);
-- Expected output: 1.75

-- 3. Unload the model when we're done with it
select infera_unload_model('linear_model');

-- 4. Check the Infera version
select infera_get_version();
````

[![Simple Demo 1](https://asciinema.org/a/745806.svg)](https://asciinema.org/a/745806)

> [!NOTE]
> After building from source, the Infera binary will be `build/release/extension/infera/infera.duckdb_extension`.
> You can load it using the `load 'build/release/extension/infera/infera.duckdb_extension';` in the DuckDB shell.
> Note that the extension binary will only work with the DuckDB version that it was built against.
> At the moment, Infera is not available as
> a [DuckDB community extension](https://duckdb.org/community_extensions/list_of_extensions).
> Nevertheless, you can still use Infera by building it from source yourself, or downloading pre-built binaries from
> the [releases page](https://github.com/CogitatorTech/infera/releases) for your platform.
> Please check the [this page](https://duckdb.org/docs/stable/extensions/installing_extensions.html) for more details on
> how to use extensions in DuckDB.

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
