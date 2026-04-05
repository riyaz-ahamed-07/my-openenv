"""Run openenv push and show full output."""
import os, sys, subprocess

os.chdir(r'C:\Users\admin\my-openenv')

env = os.environ.copy()
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(
    [sys.executable, '-m', 'openenv.cli', 'push', '--repo-id', 'DarkKnight1217/support-triage-env'],
    capture_output=True,
    encoding='utf-8',
    errors='replace',
    env=env,
    cwd=r'C:\Users\admin\my-openenv'
)

safe_out = ''.join(c if ord(c) < 128 else '?' for c in result.stdout)
safe_err = ''.join(c if ord(c) < 128 else '?' for c in result.stderr)

print("=== STDOUT ===")
for line in safe_out.split('\n'):
    print(repr(line))

print("\n=== STDERR ===")
for line in safe_err.split('\n'):
    print(repr(line))

print("\nReturn code:", result.returncode)
