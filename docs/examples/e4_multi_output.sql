-- multi-output model demonstration
.echo on
load infera;

select '## load multi-output model';
-- expects test/models/multi_output.onnx (identity: [1,4] -> [1,4])
select infera_load_model('multi_output', 'test/models/multi_output.onnx') as loaded_multi;

select '## model info';
select infera_get_model_info('multi_output') as multi_model_info; -- shows output_shape [1,4]

select '## predict_multi (returns all four values)';
select infera_predict_multi('multi_output', 1.0, 2.0, 3.0, 4.0) as multi_output_prediction; -- expect [1,2,3,4]

select '## predict (single-output API) will raise mismatch error if executed';
-- the following is intentionally commented because it raises an error:
-- select infera_predict('multi_output', 1.0, 2.0, 3.0, 4.0);

select '## cleanup';
select infera_unload_model('multi_output') as unloaded_multi;
.echo off
