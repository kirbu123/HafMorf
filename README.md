# HafMorf

Input:

```text
./data/target.Pdf
```

Result directory:

```text
./result/
```

## Launch

From this directory:

```bash
python -m src.hafmorf \
  --input <input-path> \
  --output-dir <out-dir>
```

Optional parameters:

```bash
python -m src.hafmorf \
  --input data/target.Pdf \
  --output-dir result \
  --angle-range 15 \
  --angle-step 0.1 \
  --line-scale 0.035
```

## Current Result

### Input:

![alt text](result/01_input.png)

### Oriented:

![alt text](data/02_oriented.png)

### Detected lines:

![alt text](data/03_detected_lines.png)

### Result:

![alt text](data/04_without_table_lines.png)
