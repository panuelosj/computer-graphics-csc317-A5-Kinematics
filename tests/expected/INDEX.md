# Public test data: A5-Kinematics

These are the exact arrays the public unit checks compare your
functions against — the inputs they feed in and the outputs they
expect back. Everything here is generated from
`a5_golden.npz`; nothing here is secret, and you
are meant to look at it when a check fails.

* `arrays/<name>.txt` — the values, in readable form.
* `images/<name>.png` — the same array as a viewable image, for
  the entries that are pictures.

To load an array yourself:

```python
import numpy as np
g = np.load('tests/golden/a5_golden.npz')
print(g.files)          # every available name
img = g['<name>']       # one array
```

Regenerate this folder with `python export_expected.py <assignment>` from the repository root.

| entry | what it is | values | picture |
|---|---|---|---|
| `b_indices` | shape (4,), dtype int64, range [0, 3] | [b_indices.txt](arrays/b_indices.txt) | — |
| `copy_A` | shape (12,), dtype float64, range [2, 24] | [copy_A.txt](arrays/copy_A.txt) | — |
| `cr_expected` | shape (10, 3), dtype float64, range [-10, 40] | [cr_expected.txt](arrays/cr_expected.txt) | — |
| `cr_query` | shape (10,), dtype float64, range [-0.5, 4.5] | [cr_query.txt](arrays/cr_query.txt) | — |
| `cr_times` | shape (5,), dtype float64, range [0, 4] | [cr_times.txt](arrays/cr_times.txt) | — |
| `cr_vals` | shape (5, 3), dtype float64, range [-10, 40] | [cr_vals.txt](arrays/cr_vals.txt) | — |
| `euler_inputs` | shape (5, 3), dtype float64, range [-60, 200] | [euler_inputs.txt](arrays/euler_inputs.txt) | — |
| `euler_outputs` | shape (5, 4, 4), dtype float64, range [-1, 1] | [euler_outputs.txt](arrays/euler_outputs.txt) | — |
| `fk_T` | shape (4, 4, 4), dtype float64, range [-0.904355, 1] | [fk_T.txt](arrays/fk_T.txt) | — |
| `fk_pose` | shape (4, 3), dtype float64, range [-30, 40] | [fk_pose.txt](arrays/fk_pose.txt) | — |
| `ik_b` | shape (1,), dtype int64, range [3, 3] | [ik_b.txt](arrays/ik_b.txt) | — |
| `ik_final_err` | shape (), dtype float64, range [0.0010131, 0.0010131] | [ik_final_err.txt](arrays/ik_final_err.txt) | — |
| `jac` | shape (12, 12), dtype float64, range [-0.0329027, 0.0376084] | [jac.txt](arrays/jac.txt) | — |
| `lbs_U` | shape (786, 3), dtype float64, range [-0.286034, 2.3113] | [lbs_U.txt](arrays/lbs_U.txt) | — |
| `nbones` | shape (), dtype int64, range [4, 4] | [nbones.txt](arrays/nbones.txt) | — |
| `obj_f` | shape (), dtype float64, range [4.6206, 4.6206] | [obj_f.txt](arrays/obj_f.txt) | — |
| `obj_grad` | shape (12,), dtype float64, range [-0.129846, 0] | [obj_grad.txt](arrays/obj_grad.txt) | — |
| `obj_z` | shape (12,), dtype float64, range [0, 0] | [obj_z.txt](arrays/obj_z.txt) | — |
| `proj_in` | shape (12,), dtype float64, range [-500, 500] | [proj_in.txt](arrays/proj_in.txt) | — |
| `proj_out` | shape (12,), dtype float64, range [-150, 120] | [proj_out.txt](arrays/proj_out.txt) | — |
| `tips` | shape (12,), dtype float64, range [0, 2.1548] | [tips.txt](arrays/tips.txt) | — |
| `xb0` | shape (3,), dtype float64, range [0, 2.47988] | [xb0.txt](arrays/xb0.txt) | — |
