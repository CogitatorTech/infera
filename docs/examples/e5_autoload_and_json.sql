-- autoload directory & lightweight json string inspection
.echo on
load infera;

select '## autoload non-existent directory (expect error in json)';
select infera_set_autoload_dir('nonexistent_dir___unlikely') as autoload_error_json; -- contains "error"

select '## autoload existing models directory';
select infera_set_autoload_dir('test/models') as autoload_result; -- loads linear, multi_output (if present & valid)
select infera_get_loaded_models() as loaded_models_json;          -- e.g. ["linear","multi_output"]

select '## list each loaded model info';
-- simple split-like exploration using instr; not parsing full json to avoid extension dependency
select 'has_linear' as label, instr(infera_get_loaded_models(), 'linear') > 0 as present
union all
select 'has_multi_output', instr(infera_get_loaded_models(), 'multi_output') > 0;

select '## cleanup';
select infera_unload_model('linear') as unload_linear;
select infera_unload_model('multi_output') as unload_multi_output;
.echo off
