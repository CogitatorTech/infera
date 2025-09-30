## Sample Models

| # | File                                   | Description                                                                                                                                                                 |
|---|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | [linear.onnx](linear.onnx)             | A simple linear model for end-to-end testing. Note that the model has a fixed batch size of 1 (accepts a single row).                                                       |
| 2 | [multi_output.onnx](multi_output.onnx) | A simple identity model with shape [1,4] → [1,4]. It's used to check multi-column outputs and the `infera_predict_multi` vs `infera_predict` shape mismatch error handling. |                                   |

> [!NOTE]
> All models are in ONNX format and can be used with the `infera_load_model` function.
