# Checkpoints

Place checkpoints under the following paths after downloading them from Google Drive.

| File | Target path | Size | SHA256 | Google Drive |
|---|---|---:|---|---|
| `densetnt.bin` | `advgen/pretrained/densetnt.bin` | 124868973 | `435777D3C6E1A89522F7069CA9DE980B31C99CDB23EB3A5CBECE195B08BB0081` | TODO: add link |
| `hgpo_finetuned_model_adv_best.bin` | `advgen/finetuned/hgpo_finetuned_model_adv_best.bin` | 124869787 | `D45BA27407BE4D812C5E8CBAA3C69D4680C2E97D43D67066B05C253B456B4D9C` | TODO: add link |
| `hgpo_finetuned_model_real_best.bin` | `advgen/finetuned/hgpo_finetuned_model_real_best.bin` | 124870063 | `425E11927715E624373646A4573D257C70B6F7E37065AF566342346BDFE69BB3` | TODO: add link |

Verify a downloaded file:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

for path in [
    Path("advgen/pretrained/densetnt.bin"),
    Path("advgen/finetuned/hgpo_finetuned_model_adv_best.bin"),
    Path("advgen/finetuned/hgpo_finetuned_model_real_best.bin"),
]:
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```
