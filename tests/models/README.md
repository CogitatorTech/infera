## Sample Models

| # | File                       | Description                                                                                                                                                                                                                |
|---|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | [linear.onnx](linear.onnx) | A simple linear model for end-to-end testing. Note that the model has a fixed batch size of 1 (accepts a single row). See the [test_core_functionality.sql](../sql/test_core_functionality.sql) file for an example usage. |

> [!NOTE]
> All models are in ONNX format and can be used with the `infera_load_model` function.
