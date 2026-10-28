"""
tee_run.py LOGFILE COMMAND...
Runs COMMAND, shows its output in the console and writes it to LOGFILE; returns its exit code.
"""
import subprocess
import sys

log_path, cmd = sys.argv[1], sys.argv[2:]
with open(log_path, "w", encoding="utf-8") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
sys.exit(p.wait())
