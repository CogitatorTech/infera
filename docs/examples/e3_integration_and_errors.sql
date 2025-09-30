-- integration & error handling demo
.echo on
load infera;

-- section 1: missing model behavior
select '## missing model behavior';
select infera_get_model_info('nonexistent_model') as missing_model_info;  -- returns JSON with error
select infera_unload_model('nonexistent_model') as unload_missing;        -- idempotent true

-- section 2: batch style predictions (deterministic)
select '## batch predictions';
select infera_load_model('linear', 'test/models/linear.onnx') as loaded_linear;
-- deterministic feature set (3 rows)
create or replace table features as
values
  (1, 1.0::float, 2.0::float, 3.0::float),
  (2, 0.5::float, 1.0::float, 1.5::float),
  (3, -1.0::float, 0.0::float, 2.0::float)
  ;
-- compute predictions row-wise
select column0 as id, column1 as f1, column2 as f2, column3 as f3,
       infera_predict('linear', column1, column2, column3) as prediction
from features
order by 1;
-- aggregate over the small batch
select avg(infera_predict('linear', column1, column2, column3)) as avg_prediction,
       count(*) as n
from features;

-- section 3: null feature error
select '## null feature error';
create or replace table features_with_nulls as values (1, 1.0::float, 2.0::float, null::float);
-- this will raise an error if executed directly; kept commented for demonstration
-- select infera_predict('linear', column1, column2, column3) from features_with_nulls;

-- instead, show detection:
select column0 as id,
       (column3 is null) as has_null_feature
from features_with_nulls;

-- section 4: cleanup
select '## cleanup';
drop table features;
drop table features_with_nulls;
select infera_unload_model('linear') as unloaded_linear;

.echo off
