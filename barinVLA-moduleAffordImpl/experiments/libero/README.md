# LIBERO Evaluation
## 🛠️ Installation
This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

Create virtual environment

```bash
uv venv --python 3.8 experiments/libero/.venv
source experiments/libero/.venv/bin/activate
uv pip sync experiments/libero/requirements.txt 3rd/LIBERO/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e 3rd/LIBERO
```

## 🛝  Download the Checkpoints
<table>
  <tr>
    <th>Task Suite</th>
    <th>Checkpoints</th>
  </tr>
  <tr>
    <td>LIBERO-GOAL</td>
    <td><a href="https://huggingface.co/Hume-vla/Libero-Goal-2">hume-libero-goal</a></td>
  </tr>
  <tr>
    <td>LIBERO-OBJECT</td>
    <td><a href="https://huggingface.co/Hume-vla/Libero-Object-1">hume-libero-object</a></td>
  </tr>
  <tr>
    <td>LIBERO-Spatial</td>
    <td><a href="https://huggingface.co/Hume-vla/Libero-Spatial-1">hume-libero-spatial</a></td>
  </tr>
</table>

## 🖥️  Run Evaluation
```bash
bash experiments/libero/scripts/eval_libero.sh
```


> [!NOTE]
> We provide optimal `TTS args` of each checkpoint for reproduction, you can refer to the `Optimal TTS Args` session in the corresponding model card. 
> 
> For example, you can find the `Optimal TTS Args` of `LIBERO-Goal` [here](https://huggingface.co/Hume-vla/Libero-Goal-2#optimal-tts-args)

And here are some commonly used TTS args, you can also try them in `experiments/libero/scripts/eval_libero.sh`:
```bash
# TTS args - 1
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# TTS args - 2
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.2
time_temp_lower_bound=1.0
time_temp_upper_bound=1.0

# TTS args - 3
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=2.0
time_temp_lower_bound=1.0
time_temp_upper_bound=1.0
```