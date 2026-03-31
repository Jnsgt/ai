# Input Format

The benchmark input JSON is expected to contain the following fields:

```json
{
  "input_name": "input",
  "shape": [1, 1],
  "data": [0.5]
}
```

## Fields

- `input_name`: model input tensor name
- `shape`: tensor shape used for reshaping the flat input array
- `data`: flattened numeric tensor data

All current demo inputs are stored in `examples/`.
