"""Record the machine and software environment for one benchmark stage, as one JSON file.

Usage:
    python3 scripts/record_machine.py --stage pilot --out-dir results/phase2/$(uname -m) \
        [--image onnx-docker-bench:latest]

Writes <out-dir>/machine-<stage>-<EC2 instance id, or "local">.json, so a stage that
is resumed on a replacement instance gets its own file. Standard library only.
Fields that don't apply (e.g. EC2 metadata off EC2, cpufreq on EC2) are null.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMDS = "http://169.254.169.254/latest"
errors: list[str] = []  # failed commands, kept in the JSON for diagnosis


def run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout.strip()
    except subprocess.CalledProcessError as e:
        errors.append(f"{' '.join(cmd)[:80]}: exit {e.returncode}: {e.stderr.strip()[-300:]}")
    except (OSError, subprocess.SubprocessError) as e:
        errors.append(f"{' '.join(cmd)[:80]}: {e}")
    return None


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def ec2_metadata() -> dict | None:
    """IMDSv2 (session token, then metadata). None when not on EC2."""
    try:
        token_req = urllib.request.Request(
            f"{IMDS}/api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()

        def get(path: str) -> str:
            req = urllib.request.Request(f"{IMDS}/meta-data/{path}", headers={"X-aws-ec2-metadata-token": token})
            return urllib.request.urlopen(req, timeout=2).read().decode()

        return {
            "instance_id": get("instance-id"),
            "instance_type": get("instance-type"),
            "ami_id": get("ami-id"),
            "region": get("placement/region"),
            "availability_zone": get("placement/availability-zone"),
        }
    except OSError:
        return None


def cpuinfo_model() -> dict | None:
    """Identifying /proc/cpuinfo fields of the first CPU ("model name" on x86; implementer/part on arm64)."""
    text = read("/proc/cpuinfo")
    if text is None:
        return None
    keys = ("model name", "vendor_id", "cpu family", "model", "stepping", "microcode",
            "CPU implementer", "CPU architecture", "CPU variant", "CPU part", "CPU revision",
            "flags", "Features")
    first = {}
    for line in text.splitlines():
        if not line.strip():
            if first:
                break
            continue
        key, _, value = line.partition(":")
        if key.strip() in keys:
            first[key.strip()] = value.strip()
    return first


def lscpu() -> dict | None:
    out = run("lscpu", "-J")
    if out is None:
        return None
    return {f["field"].rstrip(":"): f["data"] for f in json.loads(out)["lscpu"]}


def cgroup_version() -> str | None:
    fs = run("stat", "-fc", "%T", "/sys/fs/cgroup")
    return {"cgroup2fs": "v2", "tmpfs": "v1"}.get(fs, fs)


def cpufreq() -> dict | None:
    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    if not os.path.isdir(base):
        return None  # not exposed (typical on EC2 Nitro)
    return {k: read(f"{base}/{k}") for k in ("scaling_governor", "scaling_driver", "cpuinfo_min_freq", "cpuinfo_max_freq")}


def macos_host() -> dict | None:
    """Apple Silicon host details (the Docker VM's vCPUs run on these cores)."""
    if platform.system() != "Darwin":
        return None
    def sysctl(key: str) -> str | None:
        return run("sysctl", "-n", key)
    power = run("pmset", "-g", "batt")
    return {
        "hw_model": sysctl("hw.model"),
        "cpu_brand": sysctl("machdep.cpu.brand_string"),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "memory_bytes": sysctl("hw.memsize"),
        "macos_version": platform.mac_ver()[0],
        "power_source": power.splitlines()[0].split("'")[1] if power and "'" in power else None,
        "low_power_mode": next((line.split()[-1] for line in (run("pmset", "-g") or "").splitlines()
                                if "lowpowermode" in line), None),
    }


def docker_info() -> dict | None:
    out = run("docker", "info", "--format", "{{json .}}")
    if out is None:
        return None
    info = json.loads(out)
    keys = ("ServerVersion", "OperatingSystem", "KernelVersion", "Architecture", "NCPU", "MemTotal",
            "CgroupVersion", "CgroupDriver", "Driver")
    return {k: info.get(k) for k in keys}


def image_info(image: str) -> dict | None:
    image_id = run("docker", "image", "inspect", "--format", "{{.Id}}", image)
    if image_id is None:
        return None
    probe = (
        "import json, platform, numpy, onnxruntime as o; "
        "print(json.dumps({'ort_version': o.__version__, 'numpy_version': numpy.__version__, "
        "'python_version': platform.python_version(), 'machine': platform.machine()}))"
    )
    inside = run("docker", "run", "--rm", "--entrypoint", "python", image, "-c", probe)
    # The image ID can change on every rebuild (BuildKit attaches a provenance
    # attestation with build metadata), so also fingerprint the content: the
    # layer digests only change when the image's files do.
    layers = run("docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image)
    content = hashlib.sha256(layers.encode()).hexdigest() if layers else None
    return {"image": image, "image_id": image_id, "content_sha256": content,
            **(json.loads(inside) if inside else {"probe": "failed"})}


def git_state() -> dict | None:
    commit = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if commit is None:
        return None
    status = run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)}


def model_hashes() -> dict:
    hashes = {}
    for path in sorted((ROOT / "models").glob("*.onnx")):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hashes[path.name] = h.hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, help="stage name, e.g. pilot, input-224, input-448")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image", default="onnx-docker-bench:latest")
    args = parser.parse_args()

    ec2 = ec2_metadata()
    meminfo = read("/proc/meminfo")
    record = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": args.stage,
        "ec2": ec2,
        "macos_host": macos_host(),
        "os": {
            "pretty_name": next((line.split("=", 1)[1].strip('"') for line in (read("/etc/os-release") or "").splitlines()
                                 if line.startswith("PRETTY_NAME=")), platform.platform()),
            "kernel": platform.release(),
            "kernel_build": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "cpuinfo": cpuinfo_model(),
            "lscpu": lscpu(),
            "os_cpu_count": os.cpu_count(),
            "smt_active": read("/sys/devices/system/cpu/smt/active"),
            "cpufreq": cpufreq(),
        },
        "memory_total": next((line.split(":", 1)[1].strip() for line in (meminfo or "").splitlines()
                              if line.startswith("MemTotal:")), None),
        "transparent_hugepage": read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "cgroup_version": cgroup_version(),
        "docker": docker_info(),
        "image": image_info(args.image),
        "git": git_state(),
        "models_sha256": model_hashes(),
        "loadavg": read("/proc/loadavg") or " ".join(f"{x:.2f}" for x in os.getloadavg()),
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"machine-{args.stage}-{ec2['instance_id'] if ec2 else 'local'}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")
    image = record["image"] or {}
    print(f"  instance {ec2['instance_type'] if ec2 else 'n/a (not EC2)'}, kernel {record['os']['kernel']}, "
          f"cgroup {record['cgroup_version']}, docker {(record['docker'] or {}).get('ServerVersion')} "
          f"({(record['docker'] or {}).get('CgroupDriver')} driver), image {image.get('image_id', 'n/a')[:19]}, "
          f"ORT {image.get('ort_version')}, git {(record['git'] or {}).get('commit', 'n/a')[:12]}")
    # A benchmark host must have Docker and the image; don't let a stage start
    # with those details silently missing.
    if record["docker"] is None or "ort_version" not in image:
        raise SystemExit(f"incomplete record (docker or image info missing): {errors or 'no error captured'}")


if __name__ == "__main__":
    main()
