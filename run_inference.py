"""Run inference and log all output to a file."""
import os, sys, subprocess

env = os.environ.copy()
env['HF_SPACE_URL'] = 'https://darkknight1217-support-triage-env.hf.space'
env['API_BASE_URL'] = 'https://api.openai.com/v1'
env['MODEL_NAME'] = 'gpt-4o-mini'
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUTF8'] = '1'

result = subprocess.run(
    [sys.executable, 'inference.py'],
    capture_output=True,
    encoding='utf-8',
    errors='replace',
    env=env,
    cwd=r'C:\Users\admin\my-openenv',
    timeout=600
)

out = ''.join(c if ord(c) < 128 else '?' for c in result.stdout)
err = ''.join(c if ord(c) < 128 else '?' for c in result.stderr)

with open('inference_out.txt', 'w', encoding='ascii') as f:
    f.write("=== STDOUT ===\n")
    f.write(out)
    f.write("\n=== STDERR ===\n")
    f.write(err)
    f.write(f"\n=== Return code: {result.returncode} ===\n")

print("Done. Check inference_out.txt")
print("Last 30 lines of stdout:")
lines = out.strip().split('\n')
for l in lines[-30:]:
    print(l)
