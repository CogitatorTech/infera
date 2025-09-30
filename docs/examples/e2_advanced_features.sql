-- advanced features: remote loading & blob inference
.echo on
load infera;

-- section 1: remote model loading (small linear model hosted on github)
select '## remote model';
create or replace macro model_name() as 'remote_linear_model';
create or replace macro model_url() as 'https://github.com/CogitatorTech/infera/raw/refs/heads/main/test/models/linear.onnx';
select infera_load_model(model_name(), model_url()) as loaded_remote;          -- expect true
select instr(infera_get_loaded_models(), model_name()) > 0 as remote_listed;   -- expect true
select abs(infera_predict(model_name(), 1.0, 2.0, 3.0) - 1.75) < 1e-5 as remote_predict_ok; -- expect true
select infera_unload_model(model_name()) as remote_unloaded;                   -- expect true

-- section 2: blob inference (mobilenet example)
select '## blob inference';
-- Load a vision model from huggingface (size & load time depend on network).
select infera_load_model(
  'mobilenet',
  'https://huggingface.co/onnxmodelzoo/tf_mobilenetv3_small_075_Opset17/resolve/main/tf_mobilenetv3_small_075_Opset17.onnx'
) as mobilenet_loaded;

-- (optional) to see an error for an invalid blob size, you could run:
-- select infera_predict_from_blob('mobilenet', cast('abc' as blob)); -- would error (invalid BLOB size)

-- construct zero-filled blob of correct size: 1*224*224*3 floats * 4 bytes = 602112
with zeros as (
  select cast(repeat(chr(0), 602112) as blob) as zero_blob
)
select len(infera_predict_from_blob('mobilenet', zero_blob)) as mobilenet_blob_output_len from zeros; -- length of output list

select infera_unload_model('mobilenet') as mobilenet_unloaded;

-- optional: remote multi-output or larger models could be added similarly.
.echo off
