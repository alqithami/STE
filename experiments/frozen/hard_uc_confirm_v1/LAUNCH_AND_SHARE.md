> Anonymous reproduction copy: set `STE_SSH_KEY` to your own private-key path and
> `STE_GPU_HOST` to your own GPU host. The executed scientific files are unchanged.

# Upload, run, monitor, merge and share

Use two free L40S servers if available. Both receive the **same unmodified** package.
Server A runs collections 0–5; Server B runs 6–11. Their allocated work is 6 hours
36 minutes each, with setup, QA, test scoring and packaging additional. For a single
server, run all twelve instead; allocated work is 13 hours 12 minutes.

The following commands do not create or purchase a server. A RunPod L40S with a
working NVIDIA driver and enough disk is suitable as Server B. Copy its actual SSH
host, port and authorized private-key path from its Connect panel. Its model must
match Server A's GPU. Keep the pod running until both archive and download checks
have passed.

## 1. Mac: send the code to the existing cloud server

Download `STE_HardUC_Confirm_v1.zip` into Downloads. In a **Mac terminal**, run:

```bash
scp -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1.zip" \
  ubuntu@$STE_GPU_HOST:~/

ssh -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  ubuntu@$STE_GPU_HOST
```

Your prompt should now start with `ubuntu@…`, not your Mac username. The commands
below run on that server.

## 2. Server A: launch its six complete collections

```bash
cd "$HOME"
unzip -n STE_HardUC_Confirm_v1.zip
cd "$HOME/STE_HardUC_Confirm_v1"
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1-a"
export STE_COLLECTIONS=0,1,2,3,4,5
export STE_EXPECT_GPU_MODEL='NVIDIA L40S'
bash cloud.sh setup
bash cloud.sh start
```

Setup first reuses a compatible pinned environment from the completed UC study.
If it cannot find one, it builds a new task-local Python 3.11 environment. Do not
install the experiment into the operating-system Python 3.14 environment. If you
know the compatible interpreter's exact path, you can supply it before setup:

```bash
export STE_PYTHON='/full/path/to/the/existing/.venv/bin/python'
```

This optional path must be real; do not paste the illustrative path literally.
Do not change software versions or scientific config to get around a failing check.
On a busy GPU, the launcher refuses to start and does not stop existing work.

## 3. Mac and Server B: send the same ZIP and run the other half

In a second **Mac terminal**, use the actual connection values for Server B:

```bash
export STE_SERVER_B_HOST='PUBLIC_IP_FROM_ITS_CONNECT_PANEL'
export STE_SERVER_B_PORT='SSH_PORT_FROM_ITS_CONNECT_PANEL'
export STE_SERVER_B_USER='root'
export STE_SERVER_B_KEY="$HOME/.ssh/runpod_ed25519"

scp -P "$STE_SERVER_B_PORT" -o IdentitiesOnly=yes -i "$STE_SERVER_B_KEY" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1.zip" \
  "$STE_SERVER_B_USER@$STE_SERVER_B_HOST:/workspace/"

ssh -p "$STE_SERVER_B_PORT" -o IdentitiesOnly=yes -i "$STE_SERVER_B_KEY" \
  "$STE_SERVER_B_USER@$STE_SERVER_B_HOST"
```

Use the private key actually authorized on that server. For a second Ubuntu cloud
server, use its public IP, port 22 and user `ubuntu`, send to `~/`, and substitute
`$HOME` for `/workspace` in the next block. RunPod connection values are not the same
as the existing cloud server's IP or key.

Now in the **Server B terminal**:

```bash
cd /workspace
unzip -n STE_HardUC_Confirm_v1.zip
cd /workspace/STE_HardUC_Confirm_v1
export STE_RUN_ROOT=/workspace/ste-harduc-confirm-v1-b
export STE_COLLECTIONS=6,7,8,9,10,11
export STE_EXPECT_GPU_MODEL='NVIDIA L40S'
bash cloud.sh setup
bash cloud.sh start
```

Do not run both halves on the same GPU simultaneously. Do not use a different GPU
model for one half: the merge will reject it. For a third identical server, an
alternative balanced split is 0–3 / 4–7 / 8–11; choose one split before launch and
avoid duplicate collections.

## 4. Each server: continuously display new output

While the `STE_RUN_ROOT` variable is still set:

```bash
bash cloud.sh status
tail -n 30 -F "$STE_RUN_ROOT/results/run.log"
```

Ctrl-C stops the live log display only. It does not stop the detached worker. You can
disconnect SSH after `LAUNCH_CONFIRMED` and reconnect later. In a new session, restore
the run root and change to the code folder before checking progress.

For Server A, those reconnection commands are:

```bash
cd "$HOME/STE_HardUC_Confirm_v1"
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1-a"
bash cloud.sh status
tail -n 30 -F "$STE_RUN_ROOT/results/run.log"
```

For Server B, use `/workspace/STE_HardUC_Confirm_v1` and
`/workspace/ste-harduc-confirm-v1-b` instead.

Each shard must end with `SHARD_VERIFIED` and successfully pass:

```bash
bash cloud.sh verify
```

This means the shard can be downloaded. The complete study is still unfinished until
the two shards are merged and the final analysis and archives are verified.

If training completed but only packaging was interrupted, use:

```bash
bash cloud.sh package
bash cloud.sh verify
```

If state is `FAILED`, retain the log and all files. Share the last 80 log lines. Do
not delete the run or launch another seed to try to obtain a better result.

## 5. Mac: download both completed shard archives

The shard archive ID suffix is built from its actual collection IDs. Use wildcards
to retrieve the ZIP and `.sha256` sidecar without guessing the suffix.

In a **Mac terminal**:

```bash
mkdir -p "$HOME/Downloads/STE_HardUC_Confirm_v1_shards"

scp -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  "ubuntu@$STE_GPU_HOST:~/ste-harduc-confirm-v1-a/STE_HardUC_Confirm_v1_SHARD_*" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_shards/"

scp -P "$STE_SERVER_B_PORT" -o IdentitiesOnly=yes -i "$STE_SERVER_B_KEY" \
  "$STE_SERVER_B_USER@$STE_SERVER_B_HOST:/workspace/ste-harduc-confirm-v1-b/STE_HardUC_Confirm_v1_SHARD_*" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_shards/"

cd "$HOME/Downloads/STE_HardUC_Confirm_v1_shards"
shasum -a 256 -c ./*.sha256
```

Every checksum must say `OK`. Keep both archives; they contain full shard evidence.

## 6. Mac: send the two verified shards to Server A for merging

In the **Mac terminal**:

```bash
ssh -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  ubuntu@$STE_GPU_HOST 'mkdir -p "$HOME/ste-harduc-confirm-v1-inputs"'

scp -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_shards/"*.zip \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_shards/"*.sha256 \
  ubuntu@$STE_GPU_HOST:~/ste-harduc-confirm-v1-inputs/

ssh -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  ubuntu@$STE_GPU_HOST
```

## 7. Server A: merge, analyze and verify the complete study

In the **Server A terminal**:

```bash
cd "$HOME/STE_HardUC_Confirm_v1"
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1-merged"
unset STE_COLLECTIONS
bash cloud.sh merge "$HOME/ste-harduc-confirm-v1-inputs/"*.zip
bash cloud.sh verify
```

This uses a new root and preserves the original shard directories. It checks exactly
twelve unique complete collections, matching frozen source and compatible hardware.
The merge does not train any models and does not treat shards as statistical units.
It independently replays selected checkpoints on the same numerical GPU environment
before final packaging. Keep Server A's GPU free until this replay finishes; the
wrapper refuses a busy GPU and does not stop other work.

Completion requires **both** `COMPLETE` and a successful archive verification ending
with `ALL_ARCHIVES_VERIFIED`. A completed experiment may produce negative or mixed
findings; statistical significance is not a completion condition.

## 8. Mac: download the final evidence and share the review file

In a **Mac terminal**:

```bash
mkdir -p "$HOME/Downloads/STE_HardUC_Confirm_v1_final"

scp -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  "ubuntu@$STE_GPU_HOST:~/ste-harduc-confirm-v1-merged/STE_HardUC_Confirm_v1_*" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_final/"

cd "$HOME/Downloads/STE_HardUC_Confirm_v1_final"
shasum -a 256 -c ./*.sha256
```

Upload **`STE_HardUC_Confirm_v1_REVIEW.zip`** and its `.sha256` file to the chat.
Keep `STE_HardUC_Confirm_v1_RESULTS.zip` as the complete backup. If the review ZIP
is too large to upload, retrieve its generated parts and
`STE_HardUC_Confirm_v1_REVIEW_PARTS.json`:

```bash
scp -o IdentitiesOnly=yes -i "$STE_SSH_KEY" \
  "ubuntu@$STE_GPU_HOST:~/ste-harduc-confirm-v1-merged/*.part*" \
  "ubuntu@$STE_GPU_HOST:~/ste-harduc-confirm-v1-merged/STE_HardUC_Confirm_v1_REVIEW_PARTS.json" \
  "$HOME/Downloads/STE_HardUC_Confirm_v1_final/"
```

Upload every numbered part and the parts manifest. Do not omit a part. If you need
to reconstruct and verify a split archive on **Server A**, run:

```bash
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1-merged"
IFS= read -r STE_REPLAY_PYTHON < "$STE_RUN_ROOT/PYTHON_PATH.txt"
"$STE_REPLAY_PYTHON" "$HOME/STE_HardUC_Confirm_v1/verify_archive.py" \
  --parts "$STE_RUN_ROOT/STE_HardUC_Confirm_v1_REVIEW_PARTS.json" \
  --assemble "$STE_RUN_ROOT/reassembled_REVIEW.zip"
```

Only stop the extra server after verified evidence is downloaded.

## Single-server alternative

After uploading and extracting the same code ZIP, use a fresh root and all twelve
collections. The wrapper handles analysis and full packaging automatically:

```bash
cd "$HOME/STE_HardUC_Confirm_v1"
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1"
unset STE_COLLECTIONS
export STE_EXPECT_GPU_MODEL='NVIDIA L40S'
bash cloud.sh setup
bash cloud.sh start
tail -n 30 -F "$STE_RUN_ROOT/results/run.log"
```

After `COMPLETE`, run `bash cloud.sh verify`. Download from
`~/ste-harduc-confirm-v1/` instead of the merged directory in step 8. Do not merge or
launch the two half-shards as well as the all-collection run.
