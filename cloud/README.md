# Cloud runbook (Phase 2)

Runs the matrix from `docs/EXPERIMENT_PLAN.md` on one **c8g.2xlarge** (arm64,
Graviton4) and one **c8a.2xlarge** (x86_64, AMD EPYC Turin), Ubuntu 24.04 LTS, in
your default AWS region. All scripts run from the repo root on your machine,
except `setup.sh` and `remote_stage.sh`, which run on the VMs.

| Script | Does |
|---|---|
| `preflight.sh` | Read-only: account (last 4 digits), region, offerings, vCPU quota, AMIs, prices, cost estimate, then `provision.sh --dry-run` |
| `provision.sh [--dry-run] [arch]` | Key pair, SSH-from-your-IP security group, instances (tagged `project=onnx-docker-bench`) |
| `sync.sh [arch]` | Repo + exact model files to the VMs; verifies model sha256 on the VM |
| `setup.sh` (on VM) | Docker CE, quiet machine, cgroup v2 check, builds and verifies the image |
| `run_stage.sh pilot\|224\|448 [arch]` | Writes `machine-<stage>-<instance>.json`, then starts the stage in tmux |
| `fetch.sh [arch]` | Results back into `results/phase2/<aarch64\|x86_64>/` |
| `teardown.sh [--dry-run] [--yes] [--force]` | Terminates tagged instances, deletes key pair and group, lists leftovers |
| `ssh.sh arch [cmd]` | SSH to a VM by arch |

`[arch]` is `arm64`, `x86_64` or `all` (default).

## Safety nets

- **Auto-terminate.** Instances are launched with
  `--instance-initiated-shutdown-behavior terminate`, and user-data arms
  `shutdown -h` for `SHUTDOWN_HOURS` (default 24) after first boot, re-armed on
  every boot from a stored deadline. A forgotten instance therefore terminates by
  itself, **taking any unfetched results with it**.
- **No data loss on fetch or teardown.** `fetch.sh` only replaces a local CSV
  with a copy that has all its runs. `teardown.sh` refuses to terminate an instance
  whose results aren't all local (`--force` overrides).
- **Exact models.** `sync.sh` and `setup.sh` check `models/*.onnx` against
  `cloud/models.sha256` (the Mac's files) on both ends.
- **One stage at a time.** `remote_stage.sh` refuses to start while another stage
  or benchmark container is running, or if the stage's estimate doesn't fit before
  the auto-shutdown.
- **Secrets stay local.** The private key lives in `cloud/.state/` (gitignored,
  never synced). The VMs allow SSH only from your current IP and require IMDSv2.

## Prerequisites (once)

1. AWS CLI v2, configured for the account and region to use (`aws configure`,
   or `aws configure sso`). The principal needs EC2 (instances, key pairs,
   security groups, describe calls), `ssm:GetParameter`, `pricing:GetProducts`,
   `servicequotas:GetServiceQuota` and `tag:GetResources`.
2. The exact model files: `models/resnet50.onnx` and `models/resnet50.int8.onnx`
   must match `cloud/models.sha256`. They are gitignored; don't re-export, since
   a re-export would change the hashes.
3. Everything committed (`machine-*.json` records the git commit and whether the
   tree was dirty).

## Order of commands

```bash
cloud/preflight.sh                       # read-only checks, prices, provision --dry-run
cloud/provision.sh                       # launch both instances (~1 min until SSH is up)
cloud/sync.sh                            # code + models, sha256 verified on the VMs
cloud/ssh.sh arm64 'bash onnx-docker-bench/cloud/setup.sh'
cloud/ssh.sh x86_64 'bash onnx-docker-bench/cloud/setup.sh'

cloud/run_stage.sh pilot                 # stage 0, ~1 h
cloud/ssh.sh arm64 'tail -n 3 onnx-docker-bench/results/phase2/aarch64/pilot.log'   # progress
cloud/fetch.sh                           # then review the pilot (checklist in the plan)

cloud/run_stage.sh 224                   # stage 1, ~5 h
cloud/fetch.sh                           # then review; decide on 448 (and the worst-cell iteration floor)

cloud/run_stage.sh 448                   # stage 2, ~12 h (see "Time budget" first)
cloud/fetch.sh
cloud/teardown.sh                        # terminate, delete key pair + group, list leftovers
```

Then commit `results/phase2/`.

Watch a stage live with `cloud/ssh.sh arm64` and `tmux attach -t odb`, and detach
with `Ctrl-b d`. Never `exit` inside tmux while a stage is running: that ends the
stage. The runner resumes it if restarted, but the interrupted run is lost.

## Time budget

Pilot + 224 + 448 is ~18.5 h of machine time, inside the 24 h auto-shutdown only
if the reviews between stages are short (under ~4–5 h in total). Before the 448
stage, either:

- **Extend the deadline**, e.g. to 14 h from now, on both VMs:
  ```bash
  cloud/ssh.sh arm64 'echo $(( $(date +%s) + 14*3600 )) | sudo tee /var/lib/onnx-docker-bench-deadline && sudo systemctl restart odb-deadline'
  ```
  and the same for `x86_64`. `remote_stage.sh` refuses to start a stage that
  wouldn't finish before the deadline.
- **Or tear down after the 224 review and re-provision for 448**:
  `fetch.sh`, `teardown.sh`, then later `provision.sh`, `sync.sh` (which pushes
  `results/phase2/<arch>/` to the fresh VM so the runner sees the pilot and earlier
  runs), `setup.sh`, `run_stage.sh 448`. This is cheaper during long reviews, but
  the 448 stage then runs on a different physical host than the 224 stage; the
  `machine-*.json` files record which.

## If something goes wrong

- **SSH times out:** your public IP probably changed. Re-run `cloud/provision.sh`;
  it is idempotent and adds your new IP to the group (remove the old rule in the
  console).
- **A stage died** (instance lost, runner killed): start the same stage again.
  The runner skips runs already in the CSV and removes leftover containers.
- **An instance terminated early** (auto-shutdown): results not fetched are gone.
  Re-provision, `sync.sh` (pushes what you fetched), `setup.sh`, re-run the stage.
- **Leftovers after teardown:** `teardown.sh` lists anything still tagged
  `project=onnx-docker-bench`; terminated instances stay visible for about an hour.
