-- core functionality walkthrough (load -> info -> predict -> unload)
.echo on
load infera;

-- section 1: version & initial state
select '## version & initial state';
select infera_get_version() as version_json;           -- shows version, backend, cache dir
select infera_get_loaded_models() as initial_models;   -- expect []

-- section 2: load model and inspect
select '## load model';
select infera_load_model('linear', 'test/models/linear.onnx') as loaded;  -- expect true
select instr(infera_get_loaded_models(), 'linear') > 0 as is_listed;      -- expect 1/true
select infera_get_model_info('linear') as model_info;                     -- contains input/output shapes
select position('"input_shape"' in infera_get_model_info('linear')) > 0 as has_input_shape; -- expect true

-- section 3: prediction
-- model formula documented in tests: y = 2*f1 - 1*f2 + 0.5*f3 + 0.25
select '## prediction';
select infera_predict('linear', 1.0, 2.0, 3.0) as prediction;            -- expect 1.75
select abs(infera_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 as prediction_ok; -- expect true
select infera_predict_multi('linear', 1.0, 2.0, 3.0) as predict_multi;   -- single value inside list-like string

-- section 4: unload
select '## unload';
select infera_unload_model('linear') as unloaded;                         -- expect true
select infera_get_loaded_models() as after_unload;                        -- expect []

-- section 5: autoload (reuse existing test/models directory)
select '## autoload';
select infera_set_autoload_dir('test/models') as autoload_result;         -- loads linear (& others if added)
select instr(infera_get_loaded_models(), 'linear') > 0 as autoload_contains_linear; -- expect true
select infera_unload_model('linear') as unload_after_autoload;            -- true (idempotent)

.echo off
