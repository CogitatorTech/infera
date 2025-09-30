-- blob diagnostics: demonstrates error vs success paths for blob inference
.echo on
load infera;

select '## load mobilenet (remote)';
select infera_load_model(
  'mobilenet',
  'https://huggingface.co/onnxmodelzoo/tf_mobilenetv3_small_075_Opset17/resolve/main/tf_mobilenetv3_small_075_Opset17.onnx'
) as mobilenet_loaded;

select '## invalid blob size (commented example)';
-- running this would raise an error: Invalid BLOB size
-- select infera_predict_from_blob('mobilenet', cast('abc' as blob));

select '## allocate zero-filled blob of correct size (602112 bytes) and infer';
with zeros as (
  select cast(repeat(chr(0), 602112) as blob) as zero_blob
)
select len(infera_predict_from_blob('mobilenet', zero_blob)) as output_len from zeros;

select '## cleanup';
select infera_unload_model('mobilenet') as mobilenet_unloaded;
.echo off
