#!/usr/bin/env python
"""
Iterates over (prompt, temperature) pairs, calls llama-cli.exe with
--logits-all, then renames the generated log to
results/logits_raw/pXXXX_Ty.y.log
"""
import argparse, json, os, subprocess, tqdm, tempfile, glob, shutil, sys

LLAMA_EXE = os.path.join("bin", "llama-cli.exe")   # adjust if moved
LOG_TMP   = "tmp_llama_logs"                       # internal dir created by llama

def main():
    if not os.path.exists(LLAMA_EXE):
        sys.exit(f"❌  Cannot find {LLAMA_EXE}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0, 1.3])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--out_dir", default="results/logits_raw")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(LOG_TMP, exist_ok=True)

    with open(args.prompts, encoding="utf-8") as f:
        prompts = [
            json.loads(l)["prompt"] if l.strip().startswith("{") else l.strip()
            for l in f
        ]

    total = len(prompts) * len(args.temps)
    pbar  = tqdm.tqdm(total=total, desc="generating")

    for t in args.temps:
        for idx, prompt in enumerate(prompts):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w",
                                             encoding="utf-8") as tf:
                tf.write(prompt)
                tf_path = tf.name

            # run llama-cli, logs land in LOG_TMP/llama.log + logits_*.log
            cmd = [
                LLAMA_EXE,
                "-m", args.model,
                "--temp", str(t),
                "-n", str(args.max_tokens),
                "--file", tf_path,
                "--logits_all",
                "--logdir", LOG_TMP,
                "--n-gpu-layers", "-1",
            ]
            #result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = subprocess.run(cmd)
            os.unlink(tf_path)

            if result.returncode != 0:
                print(f"⚠️  llama-cli failed on prompt {idx}", file=sys.stderr)
                pbar.update()
                continue

            # find newest logits file
            logits_files = sorted(glob.glob(os.path.join(LOG_TMP, "logits_*.log")),
                                  key=os.path.getmtime)
            if not logits_files:
                print(f"⚠️  no logits file for prompt {idx}", file=sys.stderr)
                pbar.update()
                continue

            newest = logits_files[-1]
            dst = os.path.join(args.out_dir, f"p{idx:04d}_T{t:.1f}.log")
            shutil.move(newest, dst)
            pbar.update()

    pbar.close()
    print("✅ raw logits saved in", args.out_dir)

if __name__ == "__main__":
    main()
