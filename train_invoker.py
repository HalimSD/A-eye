import sys
import subprocess

if len(sys.argv[0]) <= 1:
  print(f"{sys.argv[0]} program_name param0=<param0> param1=<param1> ...")
  sys.exit(0)

program = sys.argv[1]
params = sys.argv[2:]

posparam = []
for param in params:
  _, val = param.split("=")
  posparam.append(val)

command = [sys.executable, program, *posparam]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
sys.stdout.write(out.decode())
sys.stdout.flush()
sys.stderr.write(err.decode())
sys.stderr.flush()
sys.exit(process.returncode)
